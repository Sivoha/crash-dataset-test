# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
import timm
import torch
from torchsummary import summary


def Model_2D_pretrained_effnet(
    dropout_val=0.2,
    out_channels=1,
):
    m = timm.create_model(
        'tf_efficientnetv2_l_in21k',
        num_classes=out_channels,
        drop_rate=dropout_val,
        pretrained=True,
    )
    m.eval()
    return m


if __name__ == '__main__':
    model = Model_2D_pretrained_effnet()
    print(summary(model, input_size=(3, 384, 384)))
    print(model)
