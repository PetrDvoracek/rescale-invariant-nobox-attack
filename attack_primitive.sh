# ssim 0.57
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion gaussian_noise --noise_std 14

# ssim 0.55
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion jpeg --quality 3

# ssim 0.59
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion blur --sigma 4.0

# ssim 0.57
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_salt_pepper --noise_ratio 0.03

# ssim 0.57
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_poisson --scale 0.3

# ssim 0.59
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_speckle --speckle_std 0.2

# ssim 0.55
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_uniform --noise_range 0.1

# ssim 0.57
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion motion_blur --kernel_size 25

# ssim 0.58
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion brightness --brightness_factor 0.4

# ssim 0.57
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion contrast --contrast_factor 0.25

# ssim 0.58
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion gamma --gamma 0.25



