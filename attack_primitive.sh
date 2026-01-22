# # SSIM

IN2K_VAL_WILD="/datasets/imagenet/val/**/*"
# OUTPUT_DIR="val_primitive_0.52_color"
OUTPUT_DIR="tmp"

# val_primitive:
# Average SSIM: 0.5714
# Average PSNR: 24.9559
# Average VIF: 0.9794
# val_primitive_ssim_52:
# Average SSIM: 0.5290
# Average PSNR: 23.9030
# Average VIF: 0.9748
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_gaussian --noise_std 17 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5532
# Average PSNR: 21.3652
# Average VIF: 0.5673
# val_primitive_ssim_52:
# Average SSIM: 0.5323
# Average PSNR: 21.1218
# Average VIF: 0.5385
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion jpeg --quality 2 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5693
# Average PSNR: 22.4822
# Average VIF: 0.1761
# val_primitive_ssim_52:
# Average SSIM: 0.5340
# Average PSNR: 21.7763
# Average VIF: 0.1273
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion blur --sigma 5.0 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5762
# Average PSNR: 20.0933
# Average VIF: 0.9467
# val_primitive_ssim_52:
# Average SSIM: 0.5264
# Average PSNR: 19.3151
# Average VIF: 0.9364
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_salt_pepper --noise_ratio 0.036 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5719
# Average PSNR: 23.6607
# Average VIF: 0.9799
# val_primitive_ssim_52:
# Average SSIM: 0.5254
# Average PSNR: 22.4162
# Average VIF: 0.9745
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_poisson --scale 0.26 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5757
# Average PSNR: 22.1792
# Average VIF: 0.9597
# val_primitive_ssim_52:
# Average SSIM: 0.5322
# Average PSNR: 20.8451
# Average VIF: 0.9482
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_speckle --speckle_std 0.2 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5733
# Average PSNR: 25.0677
# Average VIF: 0.9794
# val_primitive_ssim_52:
# Average SSIM: 0.5258
# Average PSNR: 23.8839
# Average VIF: 0.9743
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_uniform --noise_range 0.115 --num_processes 32 

# val_primitive:
# Average SSIM: 0.5784
# Average PSNR: 21.6641
# Average VIF: 0.2017
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion motion_blur --kernel_size 32 --num_processes 36 

# val_primitive:
# Average SSIM: 0.5742
# Average PSNR: 10.5825
# Average VIF: 0.1612
# val_primitive_ssim_52:
# Average SSIM: 0.5269
# Average PSNR: 10.2279
# Average VIF: 0.1417  
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion brightness --brightness_factor 0.375  --num_processes 32 

# val_primitive:
# Average SSIM: 0.5680
# Average PSNR: 14.6783
# Average VIF: 0.0466
# val_primitive_ssim_52:
# Average SSIM: 0.5142
# Average PSNR: 14.0901
# Average VIF: 0.0258
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion contrast --contrast_factor 0.16 --num_processes 32 

python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion contrast --contrast_factor 3.2 --num_processes 32

# val_primitive:
# # Average SSIM: 0.5675
# # Average PSNR: 8.0837
# # Average VIF: 0.5453
# val_primitive_ssim_52:
# Average SSIM: 0.5272
# Average PSNR: 7.4213
# Average VIF: 0.5554
python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion gamma --gamma 0.15 --num_processes 32 


# FIV
# baseline Mean VIF: 0.1664
# /mnt/nas/datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-1.15_tag-emuzc-reflect

# OUTPUT_DIR="val_primitive_fiv"



# val_primitive_ssim_52:
# Average SSIM: 0.5290
# Average PSNR: 23.9030
# Average VIF: 0.9748
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_gaussian --noise_std 190 --num_processes 32 --max_images 500

# # val_primitive:
# # Average SSIM: 0.5532
# # Average PSNR: 21.3652
# # Average VIF: 0.5673
# # val_primitive_ssim_52:
# # Average SSIM: 0.5323
# # Average PSNR: 21.1218
# # Average VIF: 0.5385
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion jpeg --quality 2 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5693
# # Average PSNR: 22.4822
# # Average VIF: 0.1761
# # val_primitive_ssim_52:
# # Average SSIM: 0.5340
# # Average PSNR: 21.7763
# # Average VIF: 0.1273
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion blur --sigma 5.0 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5762
# # Average PSNR: 20.0933
# # Average VIF: 0.9467
# # val_primitive_ssim_52:
# # Average SSIM: 0.5264
# # Average PSNR: 19.3151
# # Average VIF: 0.9364
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_salt_pepper --noise_ratio 0.036 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5719
# # Average PSNR: 23.6607
# # Average VIF: 0.9799
# # val_primitive_ssim_52:
# # Average SSIM: 0.5254
# # Average PSNR: 22.4162
# # Average VIF: 0.9745
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_poisson --scale 0.26 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5757
# # Average PSNR: 22.1792
# # Average VIF: 0.9597
# # val_primitive_ssim_52:
# # Average SSIM: 0.5322
# # Average PSNR: 20.8451
# # Average VIF: 0.9482
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_speckle --speckle_std 0.2 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5733
# # Average PSNR: 25.0677
# # Average VIF: 0.9794
# # val_primitive_ssim_52:
# # Average SSIM: 0.5258
# # Average PSNR: 23.8839
# # Average VIF: 0.9743
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion noise_uniform --noise_range 0.115 --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5784
# # Average PSNR: 21.6641
# # Average VIF: 0.2017
# # python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion motion_blur --kernel_size 32 --num_processes 36 

# # val_primitive:
# # Average SSIM: 0.5742
# # Average PSNR: 10.5825
# # Average VIF: 0.1612
# # val_primitive_ssim_52:
# # Average SSIM: 0.5269
# # Average PSNR: 10.2279
# # Average VIF: 0.1417  
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion brightness --brightness_factor 0.375  --num_processes 32 

# # val_primitive:
# # Average SSIM: 0.5680
# # Average PSNR: 14.6783
# # Average VIF: 0.0466
# # val_primitive_ssim_52:
# # Average SSIM: 0.5142
# # Average PSNR: 14.0901
# # Average VIF: 0.0258
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion contrast --contrast_factor 0.16 --num_processes 32 

# # val_primitive:
# # # Average SSIM: 0.5675
# # # Average PSNR: 8.0837
# # # Average VIF: 0.5453
# # val_primitive_ssim_52:
# # Average SSIM: 0.5272
# # Average PSNR: 7.4213
# # Average VIF: 0.5554
# python attack_primitive.py --output_dir $OUTPUT_DIR "$IN2K_VAL_WILD" --distortion gamma --gamma 0.15 --num_processes 32 


# LPIPS
# baseline: 0.4207

OUTPUT_DIR="val_primitive_lpips"

