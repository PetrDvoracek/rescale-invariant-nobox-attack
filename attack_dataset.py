import torch
import cv2
import tqdm
import click

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from pathlib import Path
import glob
import os
import numpy as np
from lib import *


def prepare_in1k(path, imagenet_classes):
    if not os.path.exists(path):
        os.mkdir(path)
    for c in imagenet_classes:
        p = os.path.join(path, c)
        if not os.path.exists(p):
            os.mkdir(p)


@click.command()
@click.argument("dataset-root")
@click.argument("ckpt")
@click.option("--tag", default="")
@click.option(
    "--dataset", type=click.Choice(["in1k", "coco", "cityscapes"]), default=""
)
@click.option("--device", default="cpu")
def main(dataset_root, ckpt, tag, dataset, device):
    # alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    alphas = [0.7]
    # alphas = [1.0]
    tile = 224
    trainee = TraineeAttacker.load_from_checkpoint(ckpt, map_location=device)
    model = AttackerInference(trainee, device, compile=False)

    def format_dirname(a, dirname=os.path.basename(dataset_root)):
        return f"{dirname}_adversarial-{a}_tag-{tag}"

    def get_paths():
        if dataset == "in1k":
            imagenet_classes = os.listdir(f"{dataset_root}/val")
            resds = f"val_{tile}x{tile}"
            prepare_in1k(os.path.join(dataset_root, resds), imagenet_classes)
            for alpha in alphas:
                advds = format_dirname(alpha)
                prepare_in1k(os.path.join(dataset_root, advds), imagenet_classes)
            return glob.glob(f"{dataset_root}/val/**/*")
        elif dataset == "coco":
            for a in alphas:
                p = os.path.join(dataset_root, format_dirname(a))
                print(p)
                if not os.path.exists(p):
                    os.mkdir(p)

            return glob.glob(f"{dataset_root}/test2017/*")
        elif dataset == "cityscapes":
            for a in alphas:
                p = dataset_root.replace("val", format_dirname(a, dirname="val"))
                print(p)
                if not os.path.exists(p):
                    os.mkdir(p)
                subdirs = os.listdir(dataset_root)
                print(subdirs)
                for subdir in subdirs:
                    sub_p = os.path.join(p, subdir)
                    if not os.path.exists(sub_p):
                        os.mkdir(sub_p)
            paths = glob.glob(f"{dataset_root}/*/*")
            return paths

    def adjust_imname(p, a):
        if dataset == "in1k":
            # TODO do not replace the val in imagename! only in dir
            return p.replace("val", format_dirname(a), 1)
        elif dataset == "coco":
            return p.replace("test2017", format_dirname(a), 1)
        elif dataset == "cityscapes":
            return p.replace(
                "/val/",
                format_dirname(a, dirname="/val") + "/",
                1,
            )

    paths = get_paths()
    for p in tqdm.tqdm(paths):
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        h_orig, w_orig, c_orig = im.shape

        h2add = tile - (h_orig % tile)
        w2add = tile - (w_orig % tile)
        x_orig_pad = np.zeros((h_orig + h2add, w_orig + w2add, c_orig), dtype=im.dtype)
        x_orig_pad[:h_orig, :w_orig, :] = im
        im = x_orig_pad

        h, w, c = im.shape
        batch = []
        for vertical in range(0, h, tile):
            for horizontal in range(0, w, tile):
                batch.append(
                    im[vertical : vertical + tile, horizontal : horizontal + tile]
                )

        batch = np.stack(batch)
        batch = batch / 255
        out_batch = model(batch)
        diff = out_batch - batch

        canvas_diff = np.zeros((h, w, c), dtype=np.float32)
        idx = 0
        for vertical in range(0, h, tile):
            for horizontal in range(0, w, tile):

                canvas_diff[
                    vertical : vertical + tile, horizontal : horizontal + tile
                ] = diff[idx]
                idx += 1
        canvas_diff = canvas_diff[:h_orig, :w_orig, :]
        im = im[:h_orig, :w_orig, :]
        for a in alphas:
            adv_im = np.clip((im.astype(np.float32) / 255) + (a * canvas_diff), 0, 1.0)
            adv_im = (adv_im * 255).astype(np.uint8)
            print(adv_im.max(), adv_im.min(), adv_im.mean())

            print(f"p {p}")
            print(f"a {a}")
            imname = adjust_imname(p, a)
            print(f"imname {imname}")
            ret = cv2.imwrite(
                imname,
                # f"./tmp{a}.jpg",
                # f"{ckpt}_example.jpg",
                cv2.cvtColor(adv_im, cv2.COLOR_RGB2BGR),
            )
            if not ret:
                raise IOError(f"Failed to write {imname}")


if __name__ == "__main__":
    main()
