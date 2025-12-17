# OLD
# for x in 0.0 0.3 0.5 0.7 1.0; do python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-emuzc/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-emuzc.csv; done
# x=0.95
# m=nszrr # ssim 0.738326315762105
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv
# x=0.91
# m=egkqy # ssim 0.7426757723778731
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv
# x=0.7
# m=emuzc # ssim 0.7367586553160712
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv
# x=0.85
# m=emuzc-reflect
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv

# NEW

# for x in 0.0 0.85 1.0 1.15 1.3; do python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-emuzc-reflect/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-emuzc-reflect.csv; done

# x=1.15
# m=emuzc-reflect # ssim 0.558124
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv
# x=1.35
# m=nszrr-reflect # ssim subset 0.5654878035415684
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv
# x=1.15 # subset ssim 0.5709681465745975
# m=egkqy-reflect  
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv

# x=0.75
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/existing_val_ags-${x}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-existing-ags.csv
# TODO
# x=1.3 # ssim subset 0.574081629255643 subset
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/existing_val_bei-${x}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-existing-bei.csv

# x=0.8
# m=lhdxy_used-reflect # ssim 0.558124
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/rescale-invariant-nobox-attack/_adversarial-${x}_tag-${m}/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-${x}_tag-${m}.csv

# python timm_validate_imagenet.py --data-dir /datasets/imagenet/kokosaci/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-x.x_tag-HIT.csv
# python timm_validate_imagenet.py --data-dir /datasets/imagenet/val_origsizexorigsize__0.7x/ --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./eval_adversarial-0.7_tag-orig.csv

# PRIITIVE SSIM 0.57
DATASET_ROOT="./val_primitive/"

for x in $(ls $DATASET_ROOT --ignore="*.csv"); do python timm_validate_imagenet.py --data-dir ${DATASET_ROOT}/${x} --torchcompile --device cuda --amp --model hardcoded_in_script --batch-size 64 --results-file ./${DATASET_ROOT}/${x}.csv; done
