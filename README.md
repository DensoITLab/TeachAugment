# TeachAugment: Data Augmentation Optimization Using Teacher Knowledge (CVPR2022, Oral)
Official Implementation of TeachAugment in PyTorch.  
arXiv: https://arxiv.org/abs/2202.12513

## Requirements
- PyTorch >= 1.9
- Torchvision >= 0.10

## Run
Training with single GPU
```
python main.py --yaml ./config/$DATASET_NAME/$MODEL
```

Training with single node multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=$N_GPUS main.py \
    --yaml ./config/$DATASET_NAME/$MODEL --dist
```

Examples
```
# Training WRN-28-10 on CIFAR-100
python main.py --yaml ./config/CIFAR100/wrn-28-10.yaml
# Training ResNet-50 on ImageNet with 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --yaml ./config/ImageNet/resnet50.yaml --dist
```
If the computational resources are limited, please try `--save_memory` option.


## Citation
If you find our project useful in your research, please cite it as follows:
```
@InProceedings{Suzuki_2022_CVPR,
    author    = {Suzuki, Teppei},
    title     = {TeachAugment: Data Augmentation Optimization Using Teacher Knowledge},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10904-10914}
}
```

## Acknowledgement
The files in ```./lib/models``` and the code in ```./lib/augmentation/imagenet_augmentation.py``` are based on the implementation of [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment).

