# %%
import albumentations as A
import torch
import matplotlib.pyplot as plt
import timm
import lightning as L
import wandb

from lib import *

for MODEL_NAME in [
    "tf_mobilenetv3_small_minimal_100",
    # "tf_mobilenetv3_small_075",
    # "tf_mobilenetv3_large_100",
    # "seresnet50",
    # "tf_efficientnet_b3",
    # "convnext_small",
    # "swin_s3_small_224",
    # "beit_base_patch16_224",
]:
    EPOCHS = 20
    BATCH_SIZE = 16
    WORKERS = 6
    # MODEL_NAME = "mobilenetv3_large_100"
    # MODEL_NAME = "seresnet50"
    # MODEL_NAME = "seresnet152"
    COCO = "/datasets/coco/coco/test2017"
    WANDB_PROJECT = "adversarial"
    WANDB_NAME = MODEL_NAME
    WANDB_GROUP = "augsim"

    L.pytorch.seed_everything(42, workers=True)

    dataset_coco = DSAugmentFactor(root=COCO, enlarge=10)

    trainloader = torch.utils.data.DataLoader(
        dataset_coco,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    model_sim = ModelSim()
    n_channels, n_images = 3, 2
    n_kernels = len(model_sim.kernels) * n_channels * n_images

    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=0, in_chans=n_kernels
    )

    trainee = Trainee(model, model_sim=model_sim, epochs=EPOCHS)
    trainee = torch.compile(trainee)
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        group=WANDB_GROUP,
        log_model=False,
    )
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        default_root_dir="./log",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath="./models",
                mode="min",
                monitor="train_loss=mae",
                filename=MODEL_NAME + "_epoch={epoch}_train_loss={train_loss}",
            ),
            L.pytorch.callbacks.LearningRateMonitor(),
        ],
        gradient_clip_val=1.0,
        precision="32",
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(trainee, trainloader)
    wandb.finish()
