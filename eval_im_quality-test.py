from lib import *


if __name__ == "__main__":
    COCO = "/datasets/coco/coco/test2017"
    MODEL_CHECKPOINT = "./models/tf_mobilenetv3_small_minimal_100_epoch=epoch=19_train_loss=train_loss=0.ckpt"
    trainee = Trainee.load_from_checkpoint(MODEL_CHECKPOINT)

    dataset_coco = DSAugmentFactor(root=COCO)
    testloader = torch.utils.data.DataLoader(
        dataset_coco,
        batch_size=16,
        num_workers=6,
        pin_memory=True,
        shuffle=False,
    )

    for im_orig, im_aug, y in testloader:
        print(im_orig.shape, im_aug.shape, y)
        break
