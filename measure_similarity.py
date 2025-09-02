import click
from pathlib import Path
import logging
import glob
import os
import cv2
import tqdm
import numpy as np

from skimage.metrics import structural_similarity as ssim


def process_image_pair(paths):
    path_corrupted, path_original = paths
    im_corrupted = cv2.imread(path_corrupted)
    im_original = cv2.imread(path_original)
    if im_corrupted.shape[0] != im_original.shape[0]:
        # some images had swapped axis...
        im_corrupted = np.swapaxes(im_corrupted, 0, 1)
    return ssim(
        cv2.cvtColor(im_corrupted, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(im_original, cv2.COLOR_BGR2GRAY),
    )


@click.command()
@click.argument("dataset_corrupted_root")
@click.argument("dataset_original_root")
@click.option("--wild", default="**/*.JPEG")
@click.option("--subset", default=None)
def main(dataset_corrupted_root, dataset_original_root, wild, subset):
    images_corrupted = glob.glob(
        os.path.join(dataset_corrupted_root, wild), recursive=True
    )
    images_original = []
    for path_corrupted in images_corrupted:
        path_original = path_corrupted.replace(
            dataset_corrupted_root, dataset_original_root
        )
        path_original = path_corrupted.replace(
            "ILSVRC2012_val_224x224_adversarial_", "ILSVRC2012_val_"
        )
        dirname = path_corrupted.split(os.sep)[-3]
        path_original = path_corrupted.replace(dirname, "val")

        if os.path.exists(path_original):
            images_original.append(path_original)
        else:
            logging.warning(
                f"Original image {path_original} not found for {path_corrupted}"
            )

    logging.info(
        f"Found {len(images_corrupted)} corrupted images and {len(images_original)} original images"
    )
    logging.info(f"Example corrupted image: {images_corrupted[0]}")
    logging.info(f"Example original image: {images_original[0]}")

    from multiprocessing import Pool, cpu_count

    # Define function at module level so it can be pickled
    if subset is not None:
        subset = int(subset)
        images_corrupted = images_corrupted[:subset]
        images_original = images_original[:subset]
    with Pool(processes=cpu_count()) as pool:
        ssim_values = list(
            tqdm.tqdm(
                pool.imap(
                    process_image_pair,
                    zip(images_corrupted, images_original),
                ),
                total=len(images_corrupted),
            )
        )

    logging.info(f"Mean SSIM: {np.mean(ssim_values)}")
    print("dataset_name SSIM")
    print(f"{dirname} {np.mean(ssim_values):.4f}")


if __name__ == "__main__":
    cv2.setNumThreads(0)
    logging.basicConfig(level=logging.INFO)
    main()
