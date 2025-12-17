# # SSIM

IN2K_VAL_WILD="/mnt/nas/datasets/imagenet/val/**/*"
OUTPUT_DIR="val_primitive"

# Average SSIM: 0.5714
# Average PSNR: 24.9559
# Average VIF: 0.9794
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_gaussian --noise_std 15 --num_processes 32

# Average SSIM: 0.5532
# Average PSNR: 21.3652
# Average VIF: 0.5673
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion jpeg --quality 3 --num_processes 32

# Average SSIM: 0.5693
# Average PSNR: 22.4822
# Average VIF: 0.1761
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion blur --sigma 4.0 --num_processes 32

# Average SSIM: 0.5762
# Average PSNR: 20.0933
# Average VIF: 0.9467
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_salt_pepper --noise_ratio 0.03 --num_processes 32

# Average SSIM: 0.5719
# Average PSNR: 23.6607
# Average VIF: 0.9799
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_poisson --scale 0.35 --num_processes 32

# Average SSIM: 0.5757
# Average PSNR: 22.1792
# Average VIF: 0.9597
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_speckle --speckle_std 0.17 --num_processes 32

# Average SSIM: 0.5733
# Average PSNR: 25.0677
# Average VIF: 0.9794
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_uniform --noise_range 0.1 --num_processes 32

# Average SSIM: 0.5784
# Average PSNR: 21.6641
# Average VIF: 0.2017
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion motion_blur --kernel_size 25 --num_processes 32

# Average SSIM: 0.5742
# Average PSNR: 10.5825
# Average VIF: 0.1612
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion brightness --brightness_factor 0.4  --num_processes 32

# Average SSIM: 0.5680
# Average PSNR: 14.6783
# Average VIF: 0.0466
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion contrast --contrast_factor 0.215 --num_processes 32

# Average SSIM: 0.5675
# Average PSNR: 8.0837
# Average VIF: 0.5453
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion gamma --gamma 0.185 --num_processes 32


# # FIV
# # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_gaussian --noise_std 14

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion jpeg --quality 3

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion blur --sigma 4.0

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_salt_pepper --noise_ratio 0.03

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_poisson --scale 0.3

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_speckle --speckle_std 0.2

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion noise_uniform --noise_range 0.1

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion motion_blur --kernel_size 25

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion brightness --brightness_factor 0.4

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion contrast --contrast_factor 0.25

# # # fiv
# python attack_primitive.py --output_dir val_primitive_vif "/mnt/nas/datasets/imagenet/val/n03045698/*" --distortion gamma --gamma 0.25