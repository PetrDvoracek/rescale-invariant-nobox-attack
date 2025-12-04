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

class GaussianNoiseDistortion(Distortion):
    """Gaussian noise distortion"""
    
    def __init__(self, noise_std: float = 14.0):
        self.noise_std = noise_std
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std/255.0, img.shape)
        return np.clip(img + noise, 0, 1)
    
    def get_name(self) -> str:
        return f"gaussian_noise"

class BlurDistortion(Distortion):
    """Gaussian blur distortion"""
    
    def __init__(self, sigma: float = 1.0):
        self.kernel_size = int(4 * sigma - 1)
        self.sigma = sigma
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for cv2 operations, then back to float
        img_uint8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (self.kernel_size, self.kernel_size), self.sigma)
        return blurred.astype(np.float32) / 255.0
    
    def get_name(self) -> str:
        return f"blur"

class JPEGCompressionDistortion(Distortion):
    """JPEG compression distortion"""
    
    def __init__(self, quality: int = 50):
        self.quality = quality
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for JPEG compression
        img_uint8 = (img * 255).astype(np.uint8)
        # Encode and decode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
        compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return compressed.astype(np.float32) / 255.0
    
    def get_name(self) -> str:
        return f"jpeg"

def process_image(img_path, distortion, output_dir, common_root):
    """Process a single image: apply distortion, save, and calculate SSIM"""
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
    ssim_score = ssim(img, distorted_img, data_range=1.0, channel_axis=2)
    return ssim_score

def get_distortion_names():
    """Get available distortion names"""
    return [x().get_name() for x in Distortion.__subclasses__()]

def get_distortion(distortion_type, **kwargs):
    """Factory function to create distortion objects"""
    distortions = {x().get_name(): x for x in Distortion.__subclasses__()}
    
    if distortion_type not in distortions:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    
    return distortions[distortion_type](**kwargs)

def main():
    parser = argparse.ArgumentParser(description='Apply distortions to images and calculate SSIM')
    parser.add_argument('glob_pattern', type=str, help='Glob pattern for input images')
    parser.add_argument('--distortion', type=str, default=get_distortion_names()[0], 
                       choices=get_distortion_names(),
                       help='Type of distortion to apply')
    
    # Gaussian noise parameters
    parser.add_argument('--noise_std', type=float, default=14.0, 
                       help='Standard deviation of Gaussian noise')
    
    # Blur parameters
    parser.add_argument('--sigma', type=float, default=1.0, 
                       help='Sigma for Gaussian blur')
    
    # JPEG compression parameters
    parser.add_argument('--quality', type=int, default=50, 
                       help='JPEG compression quality (1-100)')
    
    parser.add_argument('--output_dir', type=str, default='distorted_images', 
                       help='Root directory to save distorted images')
    parser.add_argument('--num_processes', type=int, default=None, 
                       help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()
    
    # Get all image files matching the glob pattern
    image_paths = glob.glob(args.glob_pattern)
    
    if not image_paths:
        print("No images found matching the pattern")
        return
    
    # Create distortion object based on type
    if args.distortion == 'gaussian_noise':
        distortion = get_distortion('gaussian_noise', noise_std=args.noise_std)
    elif args.distortion == 'blur':
        distortion = get_distortion('blur', sigma=args.sigma)
    elif args.distortion == 'jpeg':
        distortion = get_distortion('jpeg', quality=args.quality)
    
    # Find the common root of all input images to preserve relative structure
    common_root = os.path.commonpath(image_paths) if len(image_paths) > 1 else os.path.dirname(image_paths[0])
    
    # Set number of processes
    num_processes = args.num_processes if args.num_processes else cpu_count()
    
    # Create partial function with fixed parameters
    process_func = partial(process_image, 
                          distortion=distortion, 
                          output_dir=args.output_dir, 
                          common_root=common_root)
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        ssim_scores = list(tqdm.tqdm(pool.imap(process_func, image_paths), 
                                   total=len(image_paths), 
                                   desc=f"Processing images with {distortion.get_name()}"))
    
    # Filter out None values (failed loads)
    ssim_scores = [score for score in ssim_scores if score is not None]
    
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        print(f"\nDistortion: {distortion.get_name()}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Processed {len(ssim_scores)} images")
        print(f"Distorted images saved in: {os.path.join(args.output_dir, distortion.get_name())}")

if __name__ == "__main__":
    main()