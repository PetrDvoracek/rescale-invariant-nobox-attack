import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import timm
import lightning as L
import wandb
import skimage
import torchvision
import torchmetrics


import glob
import random
import time


class AttackerInference:
    def __init__(self, torch_model, device, compile=True):
        torch_model.eval()
        self.model = (
            torch.compile(torch_model, dynamic=True) if compile else torch_model
        )
        self.device = device

    def __call__(self, batch_np):
        batch = batch_np.swapaxes(1, -1)
        batch = torch.from_numpy(batch).to(torch.float32).to(self.device)
        with torch.no_grad():
            out_batch = self.model(batch).detach()
        return out_batch.cpu().numpy().swapaxes(1, -1)


def im2torch(im):
    return torch.from_numpy(im).swapaxes(0, -1).to(torch.float32)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    # source: https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


class DSImagenet(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass


class DSAttack(torch.utils.data.Dataset):
    def __init__(self, root, resolution, augment=None, enlarge=None):
        self.augment = augment
        fpaths = glob.glob(f"{root}/*")
        self.images = []
        print("loading images ...\n")
        for path in tqdm.tqdm(fpaths):
            im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if im.shape[0] < resolution or im.shape[1] < resolution:
                continue
            self.images.append(im)

        # make more iterations per epoch
        if enlarge is not None:
            self.images = self.images * enlarge
        print("done")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        if self.augment is not None:
            im = self.augment(image=im)["image"]

        return im


class Attacker(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        for name, module in model.named_modules():
            try:
                module.stride = (1, 1)
            except:
                print("no")

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.block = model.blocks[0]
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, 1), torch.nn.BatchNorm2d(3), torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.conv_out(x)
        return x


class TraineeAttacker(L.LightningModule):
    def __init__(self, model_attacker, model_augmentfactor):
        super().__init__()
        self.save_hyperparameters()
        # self.model_attacker = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 64, 3, bias=False, padding="same"),
        #     torch.nn.BatchNorm2d(64),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Conv2d(64, 3, 3, bias=False, padding="same"),
        #     torch.nn.BatchNorm2d(3),
        #     torch.nn.Sigmoid(),
        # )
        self.model_attacker = model_attacker

        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.model_augmentfactor = model_augmentfactor
        # self.blackbox = timm.create_model(
        #     "tf_mobilenetv3_large_100.in1k", pretrained=True
        # )
        self.blackbox = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.blackbox.eval()
        for param in self.blackbox.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.model_attacker(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                *self.model_attacker.parameters(),
            ],
            lr=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40, T_mult=2, eta_min=1e-6, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]

    def _step(self, batch, idx, stage):
        before = time.time()
        ims = batch.squeeze()
        lbls = ims.clone()

        if idx == 1:
            orig_grid = (
                torchvision.utils.make_grid(ims.detach().cpu()[:32], nrow=4) * 255
            ).to(torch.uint8)
            self.logger.log_image(f"{stage}_ims_orig", [orig_grid])

        advers_ims = self(ims)
        # print(advers_ims.max(), advers_ims.min(), advers_ims.mean())

        if idx == 1:
            adv_grid = (
                torchvision.utils.make_grid(advers_ims.detach().cpu()[:32], nrow=4)
                * 255
            ).to(torch.uint8)
            self.logger.log_image(f"{stage}_ims_adv", [adv_grid])
            diff_grid = (
                torchvision.utils.make_grid(
                    torch.abs(ims.detach().cpu()[:32] - advers_ims.detach().cpu()[:32]),
                    nrow=4,
                )
                * 255
            ).to(torch.uint8)
            self.logger.log_image(f"{stage}_diff", [diff_grid])

        loss, quality, advstrength = self.loss(lbls, advers_ims)
        # advstrength = self.loss_advers_strength(lbls, advers_ims)

        x_pad, adv_x_pad = ims, advers_ims
        # paddings = (6, 6, 6, 6, 0, 0, 0, 0)
        # x_pad = torch.nn.functional.pad(ims, paddings, mode="constant", value=0)
        # adv_x_pad = torch.nn.functional.pad(
        #     advers_ims, paddings, mode="constant", value=0
        # )
        images_ssim = (
            1.0
            - torchmetrics.functional.image.structural_similarity_index_measure(
                x_pad, adv_x_pad
            )
        )  # * 2
        images_diff = torch.abs(lbls - advers_ims).mean()
        # images_similarity = (images_ssim + images_diff) / 2
        images_similarity = images_ssim
        # images_similarity = torch.nn.BCELoss()(x_pad, adv_x_pad) * 40
        # varc = self.varc(x, adv_x) * 10
        # images_similarity =  varc + (images_ssim + images_diff) / 2

        # loss = images_similarity
        # loss = torch.max(images_similarity, feature_difference)
        # beta = 10
        # loss = (
        #     1
        #     / beta
        #     * torch.logsumexp(
        #         beta
        #         * torch.cat(
        #             (images_similarity.unsqueeze(0), advstrength.unsqueeze(0)),
        #         ),
        #         dim=0,
        #     )
        # )
        # loss = images_similarity
        # loss = quality

        self.log_dict(
            {
                f"{stage}_loss": loss.item(),
                # f"{stage}_imq": quality.item(),
                f"{stage}_imq": images_similarity.item(),
                f"{stage}_advs": advstrength.item(),
                f"{stage} exec time (s)": time.time() - before,
            },
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def loss(self, y_true, y_pred):
        quality = self.loss_image_quality(y_true, y_pred)
        strength = self.loss_advers_strength(y_true, y_pred)
        return torch.max(quality * 15, strength), quality, strength
        # return quality, quality, strength

    def loss_image_quality(self, y_true, y_pred):
        # return torch.mean(torch.abs(y_true - y_pred))
        y_pred = torch.clip(y_pred, 0, 1)
        """
        h_true = tf.histogram_fixed_width(y_true, value_range=(0,1), nbins=30)
        h_pred = tf.histogram_fixed_width(y_pred, value_range=(0,1), nbins=30)
        h_true = tf.cast(h_true, dtype="float32")
        h_pred = tf.cast(h_pred, dtype="float32")
        hist_loss = K.mean(K.abs(h_true/K.max(h_true) - h_pred/K.max(h_pred)))
        losses.append(hist_loss)
        """

        mask = (torch.abs(y_true - y_pred) > 20 / 255.0).type(torch.float32)
        y_true *= mask
        y_pred *= mask

        lpow = torch.mean(torch.pow(y_true - y_pred, 2.0))

        labs = torch.mean(torch.abs(y_true - y_pred))

        # hsl = torch.pow(rgb2hsv_torch(y_true) - rgb2hsv_torch(y_pred), 2.0)
        # hsl = torch.mean(hsl)
        #
        # y_true_ft = tf.math.real(tf.signal.fft2d(tf.cast(y_true, tf.complex64)))
        # y_pred_ft = tf.math.real(tf.signal.fft2d(tf.cast(y_pred, tf.complex64)))
        # ft        = K.abs(y_true_ft - y_pred_ft)
        # ft        = K.mean(ft)
        # losses.append(ft)

        # padding = (6, 6, 6, 6)  # (pad_left, pad_right, pad_top, pad_bottom)
        # y_true_padded = torch.nn.functional.pad(
        #     y_true, padding, "constant", 0
        # )  # The "constant" mode and 0 is for constant value
        # y_pred_padded = torch.nn.functional.pad(y_pred, padding, "constant", 0)
        # y_true_padded = y_true
        # y_pred_padded = y_pred
        # sim = 1.0 - self.ssim(y_true_padded, y_pred_padded)

        return (lpow + labs) / 2

    def loss_advers_strength(self, y_true, y_pred):
        diff = self.model_augmentfactor(y_true, y_pred)
        # diff = torch.clip(diff, 0, 1)
        return torch.mean(diff)

    def training_step(self, batch, idx):
        return self._step(batch, idx, stage="train")

    def _ims2grid(self, ims, n=32):
        return (torchvision.utils.make_grid(ims[:n], nrow=4) * 255).to(torch.uint8)

    def validation_step(self, batch, idx):
        before = time.time()
        x_orig, y_lbl = batch
        x_adv = self(x_orig)
        adv_pattern = x_adv - x_orig

        if idx == 1:
            x_orig2show = x_orig.detach().cpu()
            x_adv2show = x_adv.detach().cpu()
            orig_grid = self._ims2grid(x_orig2show)
            adv_grid = self._ims2grid(x_adv2show)
            diff_grid = self._ims2grid(torch.abs(x_orig2show - x_adv2show))

            self.logger.log_image(f"val_ims_orig", [orig_grid])
            self.logger.log_image(f"val_ims_adv", [adv_grid])
            self.logger.log_image("val_ims_diff", [diff_grid])

        log_dict = {}
        # alpha = [0.0, 1.0]
        alpha = [1.0]
        for a in alpha:
            x = x_orig + a * adv_pattern
            x = torchvision.transforms.functional.normalize(
                x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
            y_pred = self.blackbox(x)
            log_dict.update(
                {
                    f"val top1 {a}*adv": torchmetrics.functional.accuracy(
                        y_pred, y_lbl, num_classes=1000, task="multiclass", top_k=1
                    ),
                    # f"val top5 {f}*adv": torchmetrics.functional.accuracy(
                    #     y_pred, y_lbl, num_classes=1000, task="multiclass", top_k=5
                    # ),
                }
            )
        log_dict.update(
            {
                f"val exec time (s)": time.time() - before,
            },
        )

        self.log_dict(
            log_dict,
            on_epoch=True,
            prog_bar=True,
        )


class DSAugmentFactor(torch.utils.data.Dataset):
    def __init__(self, root, enlarge=None):
        self.im_paths = glob.glob(f"{root}/*")
        print("loading images ...\n")
        paths_to_remove = []
        for path in tqdm.tqdm(self.im_paths):
            im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            # filter too small images
            if im.shape[0] < 224 or im.shape[1] < 224:
                paths_to_remove.append(path)
        self.im_paths = [path for path in self.im_paths if path not in paths_to_remove]

        # make more iterations per epoch
        if enlarge is not None:
            self.im_paths = self.im_paths * enlarge
        print("done")

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx % len(self.im_paths)]
        im_orig = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

        im_orig = DSAugmentFactor.random_crop(im_orig)
        im_aug, y = DSAugmentFactor.augment_img(im_orig)

        im_orig = (torch.from_numpy(im_orig).swapaxes(0, -1) / 255).to(torch.float32)
        im_aug = (torch.from_numpy(im_aug).swapaxes(0, -1) / 255).to(torch.float32)

        return im_orig, im_aug, y

    def show_xy(self, idx):
        im_orig, im_aug, y = self[idx]
        x = x.detach()
        print(y)

        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(im_orig.swapaxes(0, -1))
        axs[1].imshow(im_aug.swapaxes(0, -1))
        plt.show()

    @staticmethod
    def random_crop(img):
        t = A.Compose(
            [
                A.RandomCrop(always_apply=True, p=1.0, height=224, width=224),
            ]
        )
        return t(image=img)["image"]

    @staticmethod
    def augment_img(img):
        img = img.astype("uint8")

        augs_all = []
        augs_all.append(
            A.ShiftScaleRotate(
                p=1.0,
                shift_limit=(0.2),
                scale_limit=(0),
                rotate_limit=(0),
                border_mode=cv2.BORDER_CONSTANT,
            )
        )
        augs_all.append(
            A.ShiftScaleRotate(
                p=1.0,
                shift_limit=(0.0),
                scale_limit=(1.0),
                rotate_limit=(0),
                border_mode=cv2.BORDER_CONSTANT,
            )
        )
        augs_all.append(
            A.ShiftScaleRotate(
                p=1.0,
                shift_limit=(0.0),
                scale_limit=(0.0),
                rotate_limit=(360),
                border_mode=cv2.BORDER_CONSTANT,
            )
        )
        augs_all.append(
            A.CoarseDropout(
                p=1.0,
                max_holes=16,
                max_height=64,
                max_width=64,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
        )
        augs_all.append(A.RandomToneCurve(p=1.0))
        augs_all.append(A.Emboss(p=1.0))
        augs_all.append(A.ChannelShuffle(p=1.0))
        augs_all.append(A.Perspective(p=1.0))
        augs_all.append(A.Blur(p=1.0, blur_limit=(11, 11)))
        augs_all.append(A.Sharpen(p=1.0))
        augs_all.append(
            A.HueSaturationValue(
                p=1.0,
                hue_shift_limit=(-30, 30),
                sat_shift_limit=(0, 0),
                val_shift_limit=(0, 0),
            )
        )
        augs_all.append(
            A.HueSaturationValue(
                p=1.0,
                hue_shift_limit=(0, 0),
                sat_shift_limit=(-40, 40),
                val_shift_limit=(0, 0),
            )
        )
        augs_all.append(
            A.HueSaturationValue(
                p=1.0,
                hue_shift_limit=(0, 0),
                sat_shift_limit=(0, 0),
                val_shift_limit=(-40, 40),
            )
        )
        augs_all.append(
            A.ImageCompression(
                p=1.0, quality_lower=50, quality_upper=80, compression_type=0
            )
        )
        augs_all.append(
            A.GaussNoise(p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0.0)
        )
        augs_all.append(A.Transpose(p=1.0))

        augs_probs = []
        augs_probs.append(0.7)
        augs_probs.append(0.7)
        augs_probs.append(0.7)
        augs_probs.append(0.8)
        augs_probs.append(0.7)
        augs_probs.append(0.4)
        augs_probs.append(0.2)
        augs_probs.append(0.7)
        augs_probs.append(0.4)
        augs_probs.append(0.3)
        augs_probs.append(0.7)
        augs_probs.append(0.7)
        augs_probs.append(0.7)
        augs_probs.append(0.7)
        augs_probs.append(0.3)
        augs_probs.append(0.5)

        augs_selected = np.zeros((len(augs_probs)))
        augs_final = []

        prob_final = random.randint(0, len(augs_probs) - 1)
        prob_act = 0
        while prob_act < prob_final:
            i = random.randint(0, len(augs_probs) - 1)
            if augs_selected[i] == True:
                continue
            if random.random() >= augs_probs[i]:
                continue
            augs_selected[i] = True
            augs_final.append(augs_all[i])
            prob_act += 1

        t = A.Compose(augs_final)
        return t(image=img)["image"].astype("float"), 1 - (len(augs_final)) / (
            len(augs_probs)
        )  #


class ModelSim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kernels = ModelSim.init_kernels()

        self.depthwise_convs = []
        for k in self.kernels:
            c = torch.nn.Conv2d(3, 3, kernel_size=len(k[1]), padding=2, groups=3)
            c.weight = torch.nn.Parameter(k.to(c.weight.dtype))
            self.depthwise_convs.append(c)
        self.depthwise_convs = torch.nn.ModuleList(self.depthwise_convs)
        self.depthwise_convs.requires_grad_(False)

        n_channels = 3
        self.bn = torch.nn.BatchNorm2d(len(self.kernels) * n_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        convoluted = []
        for c in self.depthwise_convs:
            convoluted.append(c(x))
        x = torch.cat(convoluted, dim=1)
        x = self.bn(x)
        return self.act(x)

    @staticmethod
    def init_kernels():
        kernels = []
        kx = np.array(
            [
                [8, 12, 16, 12, 8],
                [6, 9, 12, 9, 6],
                [0, 0, 0, 0, 0],
                [-6, -9, -12, -9, -6],
                [-8, -12, -16, -12, -8],
            ]
        )
        ky = cv2.rotate(kx, cv2.ROTATE_90_CLOCKWISE)
        for angle in range(0, 360, 30):
            wx = np.cos(angle / 180 * np.pi)
            wy = np.sin(angle / 180 * np.pi)
            tmp = kx * wx + ky * wy
            tmp /= np.sum(np.abs(tmp))
            kernels.append(np.expand_dims(np.stack([tmp, tmp, tmp], axis=-1), -1))
        return torch.tensor(np.array(kernels)).permute(0, 3, 4, 1, 2)

    @staticmethod
    def init_kernels2():
        kernels = []
        rad = 3
        for f_i, freq in enumerate([0.5, 0.1]):
            for a_i, angle in enumerate(range(0, 361, 45)):
                kernel = skimage.filters.gabor_kernel(
                    frequency=freq,
                    theta=np.pi / 180 * angle,
                    sigma_x=5.0,
                    sigma_y=5.0,
                    offset=0,
                    n_stds=6,
                )
                kernel = kernel.astype("float")
                kernel /= np.sum(np.abs(kernel))
                kernel = kernel - np.mean(kernel)
                cx, cy = kernel.shape[1] // 2, kernel.shape[0] // 2
                kernel = kernel[cy - rad : cy + rad + 1, cx - rad : cx + rad + 1]
                kernels.append(
                    np.expand_dims(np.stack([kernel, kernel, kernel], axis=-1), -1)
                )
        return torch.tensor(kernels).permute(0, 3, 4, 1, 2)

    def show_kernels(self, savepath=None):
        _, axs = plt.subplots(1, len(self.kernels), figsize=(40, 40))
        for k, ax in zip(self.kernels, axs):
            k = k - k.min()
            k = k / k.max()
            ax.imshow(k.squeeze().swapaxes(0, -1))
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath)


def _infer_to_get_output_shape(model, resolution=224, device="cuda"):
    # input is output from simulation network
    # which has 72 channels
    n_channels = 72
    x = torch.randn(1, n_channels, resolution, resolution).to(device)
    x = model(x)
    shape = x.shape
    return shape


class Trainee(L.LightningModule):
    def __init__(self, model, model_sim, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.epochs = epochs
        self.model = model
        self.model_sim = model_sim
        output_shape = _infer_to_get_output_shape(model, device="cuda")
        assert (
            len(output_shape) == 2
        ), f"Only 2D output is supported, got {output_shape}"
        self.fc = torch.nn.Linear(output_shape[-1], 1, bias=False)
        self.criterion = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()

    def forward(self, im_orig, im_aug):
        # with torch.no_grad(): # TODO fix this next time you train evaluator!
        #     x_orig = self.model_sim(im_orig)
        #     x_aug = self.model_sim(im_aug)
        #     x = torch.cat([x_orig, x_aug], dim=1)
        x_orig = self.model_sim(im_orig)
        x_aug = self.model_sim(im_aug)
        x = torch.cat([x_orig, x_aug], dim=1)
        x = self.model(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1
        # )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.epochs * 0.8),
                int(self.epochs * 0.9),
                int(self.epochs * 0.95),
            ],
        )
        return [optimizer], [lr_scheduler]

    def _step(self, batch, idx, stage):
        im_orig, im_aug, y = batch

        o = self(im_orig, im_aug).squeeze()
        loss = self.criterion(y, o)
        self.log_dict(
            {
                f"{stage}_loss=mae": loss.item(),
                f"{stage}_mse": self.mse(y, o).item(),
            },
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, idx, stage="train")

    def validation_step(self, batch, idx):
        self._step(batch, idx, stage="val")
