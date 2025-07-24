import albumentations as A

from albumentations.pytorch import ToTensorV2
import torch
import torchvision
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
import skimage
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # fix certificate expired

import torch._dynamo

torch._dynamo.config.suppress_errors = True

import glob
import os
import string

from lib import *


# In[3]:


BATCH_SIZE = 16
WORKERS = 8
EPOCHS = 10_000
RES = 224

MODEL_NAME = "se_resnet50"
MODEL_IMQ = "./models/tf_mobilenetv3_small_minimal_100_epoch=epoch=19_train_loss=train_loss=0.ckpt"
# MODEL_NAME = "timm-efficientnet-b3"
# MODEL_NAME = "custom-2layers"
IN1K = "/datasets/imagenet/"
COCO = "/datasets/coco/coco/test2017"
WANDB_PROJECT = "adversarial"
WANDB_NAME = MODEL_NAME
WANDB_GROUP = "attack"


class Transform(torch.nn.Module):
    """ImageClassification transformation without normalization"""

    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        interpolation=F.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.interpolation = interpolation

    def forward(self, img):
        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        # img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


aug = A.Compose(
    [
        A.ShiftScaleRotate(
            p=0.3,
            shift_limit=(0.0),
            scale_limit=(0.3),
            rotate_limit=(0),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(always_apply=True, p=1.0, height=RES, width=RES),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Perspective(p=0.5),
        A.Blur(p=0.2, blur_limit=(3, 3)),
        A.PadIfNeeded(p=1.0, min_height=RES, min_width=RES),
        A.ToFloat(max_value=255),
        # A.Normalize(),
        ToTensorV2(),
    ]
)
preproc = A.Compose(
    [
        A.PadIfNeeded(p=1.0, min_height=RES, min_width=RES),
        A.CenterCrop(width=RES, height=RES),
        A.ToFloat(max_value=255),
        # A.Normalize(),
        ToTensorV2(),
    ]
)
dataset_coco = DSAttack(COCO, resolution=RES, augment=aug)
# dataset_in1kvalsubset = DSAttack(
#     os.path.join(IN1K, "val", "n01440764"), augment=preproc, resolution=RES
# )
dataset_in1kval = torchvision.datasets.ImageNet(
    root=IN1K,
    split="val",
    # transform=lambda x: preproc(image=np.array(x))["image"],
    transform=Transform(crop_size=RES, resize_size=RES),
)
# dataset_coco = dataset_in1kvalsubset

# In[5]:


loader_coco = torch.utils.data.DataLoader(
    dataset_coco,
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    pin_memory=True,
    shuffle=True,
    # persistent_workers=True,
)
loader_in1kvalsubset = torch.utils.data.DataLoader(
    # dataset_in1kvalsubset,
    dataset_in1kval,
    batch_size=BATCH_SIZE * 6,
    num_workers=WORKERS,
    pin_memory=True,
    shuffle=False,
    # persistent_workers=True,
)

model_augmentfactor = Trainee.load_from_checkpoint(
    # "/workspaces/pedro/playground/imagenet_attack/good_models/quality_estimator.ckpt"
    MODEL_IMQ
)
for param in model_augmentfactor.model.parameters():
    param.requires_grad_(False)


model_attacker = smp.Unet(
    MODEL_NAME, classes=3, activation="sigmoid", encoder_weights=None
)
# base_attacker = timm.create_model("tf_efficientnet_b0", global_pool="", num_classes=0)
# model_attacker = Attacker(base_attacker)


trainee = TraineeAttacker(
    model_attacker=model_attacker, model_augmentfactor=model_augmentfactor
)
# trainee = torch.compile(trainee, dynamic=True, mode="reduce-overhead")

wandb.finish()
wandb_logger = L.pytorch.loggers.WandbLogger(
    project=WANDB_PROJECT,
    name=WANDB_NAME,
    group=WANDB_GROUP,
    log_model=False,
)
id = "".join(random.choices(string.ascii_lowercase, k=5))
trainer = L.Trainer(
    max_epochs=EPOCHS,
    logger=wandb_logger,
    default_root_dir="./log",
    callbacks=[
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=f"./models/{id}",
            save_top_k=-1,  # save all
            every_n_train_steps=2535,
        ),
        L.pytorch.callbacks.LearningRateMonitor(),
    ],
    gradient_clip_val=1.0,
    precision="16-mixed",
    log_every_n_steps=10,
    # check_val_every_n_epoch=1,
    # deterministic=True,
)
trainer.fit(
    trainee, train_dataloaders=loader_coco, val_dataloaders=loader_in1kvalsubset
)
wandb.finish()

# %%
