import torch
import torch.nn as nn
# import unet_model
from .unet_model import UNet
import torchvision as tv
import pdb


class Net(nn.Module):
    def __init__(self, unet, resnet):
        super(Net, self).__init__()
        self.unet = unet
        self.resnet = resnet

    def forward(self, input):
        seg_logits = self.unet(input)
        hybrid_input = torch.cat([seg_logits, input], dim=1)
        logits = self.resnet(hybrid_input)
        return seg_logits, logits


def make_net(name):
    if name == "ResUNet":
        unet = UNet(n_channels=1, n_classes=1)
        resnet = tv.models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net = Net(unet, resnet)
    elif name == "ResNet":
        net = tv.models.resnet50(pretrained=False)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif name == "UNet":
        net = UNet(n_channels=1, n_classes=1)
    return net


def freeze_net(net):
    for (name, param) in net.named_parameters():
        param.requires_grad = False
    return net


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def freeze_bn(net):
    for (name, layer) in net.named_modules():
        if "bn" in name:
            layer.track_running_stats = False
            layer.eval()






