# RedNet-tensorflow
Unofficial implementation of RedNet architecture described in paper ['RedNet: Residual Encoder-Decoder Network for indoor RGB-D Semantic Segmentation'](https://arxiv.org/abs/1806.01054). It's a model for "Semantic Segmentation", but fixed it for "Super Resolution" by modifying the output layer's activation function.

## Requirements
- python 3.7+
- tensorflow 2.0.0+

## Train

```python
python train.py \
        --num_layers 15 \
        --dataset dataset/bsd_images \
        --num_epoch 1000 \
        --train_batch_size 16 \
        --valid_batch_size 1
```

## TensorBoard


## License
Apache License 2.0
