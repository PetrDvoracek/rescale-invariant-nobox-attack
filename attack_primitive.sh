# # SSIM
# # Average SSIM: 0.5945
# # Average PSNR: 25.5380
# # Average VIF: 0.9816
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion noise_gaussian --noise_std 14

# # Average SSIM: 0.5532
# # Average PSNR: 21.3652
# # Average VIF: 0.5673
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion jpeg --quality 3

# Average SSIM: 0.5693 
# Average PSNR: 22.4822
# Average VIF: 0.1761
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --num_processes 8 --distortion blur --sigma 4.0

# Average SSIM: 0.5762
# Average PSNR: 20.0934
# Average VIF: 0.9467
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion noise_salt_pepper --noise_ratio 0.03

# Average SSIM: 0.5481 
# Average PSNR: 23.0145
# Average VIF: 0.9773
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion noise_poisson --scale 0.3

# Average SSIM: 0.5328 
# Average PSNR: 20.8451
# Average VIF: 0.9482
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion noise_speckle --speckle_std 0.2

# Average SSIM: 0.5733
# Average PSNR: 25.0677
# Average VIF: 0.9794
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion noise_uniform --noise_range 0.1

# Average SSIM: 0.5784
# Average PSNR: 21.6641
# Average VIF: 0.2017
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion motion_blur --kernel_size 25

# Average SSIM: 0.5742
# Average PSNR: 10.5825
# Average VIF: 0.1612
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion brightness --brightness_factor 0.4

# Average SSIM: 0.6011
# Average PSNR: 15.0744
# Average VIF: 0.0630
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion contrast --contrast_factor 0.25

# Average SSIM: 0.6314
# Average PSNR: 9.3301
# Average VIF: 0.5682
python attack_primitive.py --output_dir val_primitive "/mnt/nas/datasets/imagenet/val/**/*" --distortion gamma --gamma 0.25



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