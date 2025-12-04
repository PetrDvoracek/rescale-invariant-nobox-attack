import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse
import os
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(img_path, noise_std, output_dir, common_root):
    """Process a single image: add noise, save, and calculate SSIM"""
    # Load image (BGR format)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std/255.0, img.shape)
    noisy_img = np.clip(img + noise, 0, 1)
    
    # Preserve directory structure
    relative_path = os.path.relpath(img_path, common_root)
    output_path = os.path.join(output_dir, relative_path)
    
    # Create subdirectories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert back to [0, 255] for saving
    noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy_img_uint8)
    
    # Calculate SSIM for color images
    ssim_score = ssim(img, noisy_img, data_range=1.0, channel_axis=2)
    return ssim_score

def main():
    parser = argparse.ArgumentParser(description='Calculate SSIM between original and noisy images')
    parser.add_argument('glob_pattern', type=str, help='Glob pattern for input images')
    parser.add_argument('--noise_std', type=float, default=14.0, help='Standard deviation of Gaussian noise, this value should give ssim 0.56')
    parser.add_argument('--output_dir', type=str, default='noisy_images', help='Root directory to save noisy images')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()
    
    # Get all image files matching the glob pattern
    image_paths = glob.glob(args.glob_pattern)
    
    if not image_paths:
        print("No images found matching the pattern")
        return
    
    # Find the common root of all input images to preserve relative structure
    common_root = os.path.commonpath(image_paths) if len(image_paths) > 1 else os.path.dirname(image_paths[0])
    
    # Set number of processes
    num_processes = args.num_processes if args.num_processes else cpu_count()
    
    # Create partial function with fixed parameters
    process_func = partial(process_image, 
                          noise_std=args.noise_std, 
                          output_dir=args.output_dir, 
                          common_root=common_root)
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        ssim_scores = list(tqdm.tqdm(pool.imap(process_func, image_paths), 
                                   total=len(image_paths), 
                                   desc="Processing images"))
    
    # Filter out None values (failed loads)
    ssim_scores = [score for score in ssim_scores if score is not None]
    
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        print(f"\nAverage SSIM: {avg_ssim:.4f}")
        print(f"Processed {len(ssim_scores)} images")
        print(f"Noisy images saved in: {args.output_dir}")

if __name__ == "__main__":
    main()