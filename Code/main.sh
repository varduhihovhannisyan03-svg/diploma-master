torchrun --nproc_per_node=3 main.py --epochs 450 --batch-size 320 --model-ema --lr 0.048 --lr-warmup-init 1e-6 --weight-decay 1e-5 --model-ema-decay 0.9999 --interpolation bicubic --random-erase 0.2
