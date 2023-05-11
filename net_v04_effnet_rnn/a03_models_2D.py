# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 1
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
import timm
import torch
from torchsummary import summary
from torch import nn


class NetRNN(nn.Module):
    def __init__(self, params_model):
        super(NetRNN, self).__init__()
        num_classes = params_model["num_classes"]
        drop_rate = params_model["drop_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        base_model = timm.create_model(
            'densenet121',
            num_classes=num_classes,
            drop_rate=drop_rate,
            pretrained=pretrained,
        )
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Identity()
        self.baseModel = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        # print(x[:, ii].shape)
        y = self.baseModel((x[:, ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out


def Model_2D_with_RNN_densenet(params_model):
    m = NetRNN(params_model)
    m.eval()
    return m


if __name__ == '__main__':
    params_model = {
        "num_classes": 1,
        "drop_rate": 0.1,
        "pretrained": True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 256,
    }
    model = Model_2D_with_RNN_densenet(params_model)
    # print(summary(model, input_size=(30, 3, 224, 224)))
    data = torch.from_numpy(np.zeros((16, 30, 3, 224, 224), dtype=np.float32))
    with torch.no_grad():
        res = model(data)
        print(res.cpu().numpy())
    exit()
    print(model)
