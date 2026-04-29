## Implementation of [EfficientNet B0-B7](https://arxiv.org/abs/1905.11946) in PyTorch


| Model Name      | Num Params | Acc@1  | Acc@5  | Input size | Weights |
|-----------------|------------|--------|--------|------------|---------|
| EfficientNet-b0 | 5.2M       | 76.732 | 93.216 | 224x224x3  | ✗       |
| EfficientNet-b1 | 7.8M       | 78.700 | 94.372 | 240x240x3  | ✔       |
| EfficientNet-b2 | 9.1M       |        |        | 260x260x3  | ✗       |
| EfficientNet-b3 | 12.2M      |        |        | 300x300x3  | ✗       |
| EfficientNet-b4 | 19.3M      |        |        | 380x380x3  | ✗       |
| EfficientNet-b5 | 30.3M      |        |        | 456x456x3  | ✗       |
| EfficientNet-b6 | 43M        |        |        | 528x528x3  | ✗       |
| EfficientNet-b7 | 66.3M      |        |        | 600x600x3  | ✗       |


### Dataset

Specify the IMAGENET data folder in the `main.py` file.

``` python
parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")
```

IMAGENET folder structure:

```
├── IMAGENET 
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ....
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
```

#### Augmentation:

Random Augmentation [`RandomAugment`](efficientnet/utils/augment.py) in `efficientnet/utils/augment.py` is used as an
augmentation. To check the random augmentation run the `augment.py` file. Interpolation mode for Random Augmentation
randomly chosen from `BILINEAR` and `BICUBIC`. For resizing the input image `BICUBIC` interpolation is used.

### Train

Distributed Data Parallel - `bash main.sh`
`main.sh`:

```
torchrun --nproc_per_node=$num_gpu main.py --epochs 450 --batch-size 320 --model-ema --lr 0.048 --lr-warmup-init 1e-6 --weight-decay 1e-5 --model-ema-decay 0.9999 --interpolation bicubic --random-erase 0.2
```

Data Parallel (without DDP, not recommended) - `python main.py`

To resume the training add `--resume @path_to_checkpoint` to `main.sh`, e.g: `--resume weights/last.ckpt`

### Evaluation

```
torchrun --nproc_per_node=$num_gpu main.py --epochs 450 --batch-size 320 --model-ema --lr 0.048 --lr-warmup-init 1e-6 --weight-decay 1e-5 --model-ema-decay 0.9999 --interpolation bicubic --random-erase 0.2 --resume weights/last.ckpt --test
```
or
```
python main.py --test
```
