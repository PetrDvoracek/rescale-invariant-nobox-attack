import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import argparse
import os
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from abc import ABC, abstractmethod
from sewar.full_ref import vifp, msssim

cv2.setNumThreads(0)  # Disable OpenCV multithreading to avoid oversubscription


class Distortion(ABC):
    """Abstract base class for image distortions"""

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply distortion to image. img is normalized to [0, 1]"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return distortion name for output directory"""
        pass

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add distortion-specific arguments to the argument parser"""
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace):
        """Create distortion instance from parsed arguments"""
        pass


class NoiseGaussianDistortion(Distortion):
    """Gaussian noise distortion"""

    def __init__(self, noise_std: float = 17.0):
        self.noise_std = noise_std

    def apply(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std / 255.0, img.shape)
        return np.clip(img + noise, 0, 1)

    def get_name(self) -> str:
        return "noise_gaussian"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--noise_std",
            type=float,
            default=14.0,
            help="[noise_gaussian] Standard deviation of Gaussian noise",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(noise_std=args.noise_std)


class BlurDistortion(Distortion):
    """Gaussian blur distortion"""

    def __init__(self, sigma: float = 5.0):
        self.sigma = sigma
        self.kernel_size = int(4 * self.sigma - 1)

    def apply(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for cv2 operations, then back to float
        img_uint8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(
            img_uint8, (self.kernel_size, self.kernel_size), self.sigma
        )
        return blurred.astype(np.float32) / 255.0

    def get_name(self) -> str:
        return "blur"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sigma", type=float, default=1.0, help="[blur] Sigma for Gaussian blur"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(sigma=args.sigma)


class JPEGCompressionDistortion(Distortion):
    """JPEG compression distortion"""

    def __init__(self, quality: int = 2):
        self.quality = quality

    def apply(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for JPEG compression
        img_uint8 = (img * 255).astype(np.uint8)
        # Encode and decode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encimg = cv2.imencode(".jpg", img_uint8, encode_param)
        compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return compressed.astype(np.float32) / 255.0

    def get_name(self) -> str:
        return "jpeg"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--quality",
            type=int,
            default=50,
            help="[jpeg] JPEG compression quality (1-100)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(quality=args.quality)


class NoiseSaltPepperDistortion(Distortion):
    """Salt and pepper noise distortion"""

    def __init__(self, noise_ratio: float = 0.036):
        self.noise_ratio = noise_ratio

    def apply(self, img: np.ndarray) -> np.ndarray:
        noisy = img.copy()
        total_pixels = img.size

        # Salt noise (white pixels)
        num_salt = int(self.noise_ratio * total_pixels * 0.5)
        coords = [np.random.randint(0, i, num_salt) for i in img.shape]
        noisy[coords[0], coords[1], coords[2]] = 1.0

        # Pepper noise (black pixels)
        num_pepper = int(self.noise_ratio * total_pixels * 0.5)
        coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
        noisy[coords[0], coords[1], coords[2]] = 0.0

        return noisy

    def get_name(self) -> str:
        return "noise_salt_pepper"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--noise_ratio",
            type=float,
            default=0.05,
            help="[noise_salt_pepper] Ratio of pixels affected by salt & pepper noise (0-1)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(noise_ratio=args.noise_ratio)


class NoisePoissonDistortion(Distortion):
    """Poisson noise distortion"""

    def __init__(self, scale: float = 0.26):
        self.scale = scale

    def apply(self, img: np.ndarray) -> np.ndarray:
        # Scale image and apply Poisson noise
        scaled_img = img * self.scale
        noisy = np.random.poisson(scaled_img * 255) / 255.0 / self.scale
        return np.clip(noisy, 0, 1)

    def get_name(self) -> str:
        return "noise_poisson"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--scale",
            type=float,
            default=1.0,
            help="[noise_poisson] Scale factor for Poisson noise",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(scale=args.scale)


class NoiseSpeckleDistortion(Distortion):
    """Speckle (multiplicative) noise distortion"""

    def __init__(self, speckle_std: float = 0.2):
        self.noise_std = speckle_std

    def apply(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(1.0, self.noise_std, img.shape)
        return np.clip(img * noise, 0, 1)

    def get_name(self) -> str:
        return "noise_speckle"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--speckle_std",
            type=float,
            default=0.1,
            help="[noise_speckle] Standard deviation for speckle (multiplicative) noise",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(speckle_std=args.speckle_std)


class NoiseUniformDistortion(Distortion):
    """Uniform noise distortion"""

    def __init__(self, noise_range: float = 0.115):
        self.noise_range = noise_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(-self.noise_range, self.noise_range, img.shape)
        return np.clip(img + noise, 0, 1)

    def get_name(self) -> str:
        return "noise_uniform"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--noise_range",
            type=float,
            default=0.1,
            help="[noise_uniform] Range of uniform noise",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(noise_range=args.noise_range)


class MotionBlurDistortion(Distortion):
    """Motion blur distortion"""

    def __init__(self, kernel_size: int = 15, angle: float = 0.0):
        self.kernel_size = kernel_size
        self.angle = angle

    def apply(self, img: np.ndarray) -> np.ndarray:
        # Create motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2

        # Create line kernel
        for i in range(self.kernel_size):
            kernel[center, i] = 1
        kernel = kernel / self.kernel_size

        # Rotate kernel by angle
        M = cv2.getRotationMatrix2D((center, center), self.angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (self.kernel_size, self.kernel_size))

        # Apply kernel to image
        img_uint8 = (img * 255).astype(np.uint8)
        blurred = cv2.filter2D(img_uint8, -1, kernel)
        return blurred.astype(np.float32) / 255.0

    def get_name(self) -> str:
        return "motion_blur"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--kernel_size",
            type=int,
            default=15,
            help="[motion_blur] Size of motion blur kernel",
        )
        parser.add_argument(
            "--angle",
            type=float,
            default=0.0,
            help="[motion_blur] Angle of motion blur in degrees",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(kernel_size=args.kernel_size, angle=args.angle)


class BrightnessDistortion(Distortion):
    """Brightness adjustment distortion"""

    def __init__(self, brightness_factor: float = 0.1):
        self.brightness_factor = brightness_factor

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.clip(img * self.brightness_factor, 0, 1)

    def get_name(self) -> str:
        return "brightness"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--brightness_factor",
            type=float,
            default=1.2,
            help="[brightness] Brightness adjustment factor",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(brightness_factor=args.brightness_factor)


class ContrastDistortion(Distortion):
    """Contrast adjustment distortion"""

    def __init__(self, contrast_factor: float = 0.1):
        self.contrast_factor = contrast_factor

    def apply(self, img: np.ndarray) -> np.ndarray:
        # Adjust contrast by scaling around mean
        mean_val = np.mean(img)
        return np.clip((img - mean_val) * self.contrast_factor + mean_val, 0, 1)

    def get_name(self) -> str:
        return "contrast"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--contrast_factor",
            type=float,
            default=1.5,
            help="[contrast] Contrast adjustment factor",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(contrast_factor=args.contrast_factor)


class GammaDistortion(Distortion):
    """Gamma correction distortion"""

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.power(img, self.gamma)

    def get_name(self) -> str:
        return "gamma"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gamma", type=float, default=0.5, help="[gamma] Gamma correction value"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(gamma=args.gamma)


def process_image(img_path, distortion, output_dir, common_root):
    """Process a single image: apply distortion, save, and calculate SSIM, PSNR, VIF, FSIM, and MS-SSIM"""
    # Load image (BGR format)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Apply distortion
    distorted_img = distortion.apply(img)

    # Preserve directory structure
    relative_path = os.path.relpath(img_path, common_root)
    distortion_dir = os.path.join(output_dir, distortion.get_name())
    output_path = os.path.join(distortion_dir, relative_path)

    # Create subdirectories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert back to [0, 255] for saving
    distorted_img_uint8 = (distorted_img * 255).astype(np.uint8)
    cv2.imwrite(output_path, distorted_img_uint8)

    # Calculate SSIM for color images
    # TODO pregenerovat tak aby sedelo se skriptem measure_similarity.py
    # nastavit SSIM pri porovnavani uint8
    ssim_score = ssim(
        (img * 255).astype(np.uint8),
        distorted_img_uint8,
        data_range=255,
        channel_axis=2,
    )

    # Calculate PSNR
    mse = np.mean((img - distorted_img) ** 2)
    psnr_score = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")

    # Calculate VIF
    vif_score = vifp(img, distorted_img.astype(np.float32))

    # # Calculate FSIM
    # fsim_score = fsim(img, distorted_img)

    # # Calculate MS-SSIM
    # msssim_score = msssim(
    #     (img * 255).astype(np.uint8), (distorted_img * 255).astype(np.uint8)
    # )
    msssim_score = 0
    return ssim_score, psnr_score, vif_score, 0, msssim_score


def get_all_distortion_subclasses():
    """Recursively get all subclasses of Distortion"""

    def _get_subclasses(cls):
        subclasses = cls.__subclasses__()
        for subclass in subclasses.copy():
            subclasses.extend(_get_subclasses(subclass))
        return subclasses

    return _get_subclasses(Distortion)


def get_distortion_names():
    """Get available distortion names by discovering all Distortion subclasses"""
    return [cls().get_name() for cls in get_all_distortion_subclasses()]


def get_distortion_classes():
    """Get mapping of distortion names to classes by discovering all Distortion subclasses"""
    return {cls().get_name(): cls for cls in get_all_distortion_subclasses()}


def main():
    # Discover available distortions
    distortion_classes = get_distortion_classes()
    distortion_names = list(distortion_classes.keys())

    parser = argparse.ArgumentParser(
        description="Apply distortions to images and calculate SSIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available distortions: {', '.join(distortion_names)}

Distortion-specific parameters can be found by checking the help for each distortion class.
        """,
    )

    parser.add_argument("glob_pattern", type=str, help="Glob pattern for input images")
    parser.add_argument(
        "--distortion",
        type=str,
        default=distortion_names[0] if distortion_names else None,
        choices=distortion_names,
        help="Type of distortion to apply",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="distorted_images",
        help="Root directory to save distorted images",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all images)",
    )

    # Let each distortion class add its own arguments
    for distortion_cls in distortion_classes.values():
        distortion_cls.add_arguments(parser)

    args = parser.parse_args()

    # Get all image files matching the glob pattern
    image_paths = glob.glob(args.glob_pattern)
    if args.max_images is not None:
        np.random.seed(42)
        image_paths = np.random.choice(
            image_paths, size=args.max_images, replace=False
        ).tolist()

    if not image_paths:
        print("No images found matching the pattern")
        return

    # Get the distortion class and create instance
    distortion_cls = distortion_classes.get(args.distortion)
    if distortion_cls is None:
        print(f"Unknown distortion: {args.distortion}")
        return

    # Create distortion instance from arguments
    distortion = distortion_cls.from_args(args)

    print(f"Using distortion: {args.distortion}")

    # Find the common root of all input images to preserve relative structure
    common_root = (
        os.path.commonpath(image_paths)
        if len(image_paths) > 1
        else os.path.dirname(image_paths[0])
    )

    # Set number of processes
    num_processes = args.num_processes if args.num_processes else cpu_count()

    # Create partial function with fixed parameters
    process_func = partial(
        process_image,
        distortion=distortion,
        output_dir=args.output_dir,
        common_root=common_root,
    )

    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(process_func, image_paths, chunksize=4),
                total=len(image_paths),
                desc=f"Processing images with {distortion.get_name()}",
            )
        )

    # Filter out None values (failed loads)
    results = [result for result in results if result is not None]

    if results:
        ssim_scores, psnr_scores, vif_scores, fsim_scores, msssim_scores = zip(*results)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        avg_vif = np.mean(vif_scores)
        avg_fsim = np.mean(fsim_scores)
        avg_msssim = np.mean(msssim_scores)
        print(f"\nDistortion: {distortion.get_name()}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average VIF: {avg_vif:.4f}")
        print(f"Average FSIM: {avg_fsim:.4f}")
        print(f"Average MS-SSIM: {avg_msssim:.4f}")
        print(f"Processed {len(results)} images")
        print(
            f"Distorted images saved in: {os.path.join(args.output_dir, distortion.get_name())}"
        )


if __name__ == "__main__":
    main()
