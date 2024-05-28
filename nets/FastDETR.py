# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
from .backbonelite import Backbone, FrozenBatchNorm2d
#from .v5backbone import Backbone
from .resnet import ResNet
from . import ops
from .ops import NestedTensor, nested_tensor_from_tensor_list, unused
from .agentencode import agentEncoder
from .decode import FastDETRTransformer

from .mobilenetv2 import mobilenet_v2
from .mobilenetv3 import mobilenet_v3
from .VAN import van_b0
from .mobilenone import mobileone




class MobileNetV2(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        #print(out3.shape, out4.shape, out5.shape)
        return [out3, out4, out5]

class MobileNetV3(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return [out3, out4, out5]
class VAN(nn.Module):
    def __init__(self, pretrained = False):
        super(VAN, self).__init__()
        self.model = van_b0(pretrained=pretrained)

    def forward(self, x):
        x = self.model.forward_features(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, backbone,  num_classes,   aux_loss, flag_train=True,
                 pretrained=False,
                 multi_scale=[640, 640, 640, 672, 704, 736, 768, 800],
                ):
        super().__init__()

        self.num_classes = num_classes
        if backbone== "cspdarknet":

            self.backbone = Backbone(32, 1, 1,  pretrained=pretrained)
            self.encode = agentEncoder(hidden_dim=256, in_channels=[128, 256, 512])
            self.transformer = FastDETRTransformer(num_classes=self.num_classes, hidden_dim=256,
                                                 feat_channels=[256, 256, 256], num_decoder_layers=3, aux_loss=aux_loss)

        elif backbone== "resnet18":
            self.backbone = ResNet(depth=18)
            self.encode = agentEncoder(hidden_dim=256, in_channels=[128, 256, 512])
            self.transformer = FastDETRTransformer(num_classes=num_classes,hidden_dim=256, feat_channels=[256,256,256], num_decoder_layers=3,aux_loss=aux_loss)


        elif backbone=="mobilenetv2":
            self.backbone = MobileNetV2(pretrained=pretrained)

            self.encode = agentEncoder(hidden_dim=256, in_channels=[24, 72, 240])
            self.transformer = FastDETRTransformer(num_classes=self.num_classes, hidden_dim=256,
                                                 feat_channels=[256, 256, 256 ], num_decoder_layers=3, aux_loss=aux_loss)

        elif backbone=="mobilenetv3":
            self.backbone = MobileNetV3(pretrained=pretrained)

            self.encode = agentEncoder(hidden_dim=256, in_channels=[24, 56, 80])
            self.transformer = FastDETRTransformer(num_classes=self.num_classes, hidden_dim=256,
                                                 feat_channels=[256, 256, 256], num_decoder_layers=3, aux_loss=aux_loss)

        elif backbone=="VAN":
            self.backbone = VAN(pretrained=pretrained)

            # self.encode = agentEncoder(hidden_dim=256, in_channels=[64, 128, 256])
            # self.transformer = RTDETRTransformer(num_classes=self.num_classes, hidden_dim=256,
            #                                      feat_channels=[256,256,256 ], num_decoder_layers=3, aux_loss=aux_loss)
            self.encode = agentEncoder(hidden_dim=128, in_channels=[64, 160, 256])
            self.transformer = FastDETRTransformer(num_classes=self.num_classes, hidden_dim=128,
                                                 feat_channels=[128,], num_decoder_layers=2, aux_loss=aux_loss)
        elif backbone == "mobileone":
            self.backbone = mobileone(variant="s0", inference_mode=False)
            self.encode = agentEncoder(hidden_dim=128, in_channels=[128, 256, 1024])
            self.transformer = FastDETRTransformer(num_classes=num_classes, hidden_dim=128,
                                                 feat_channels=[128, 128, 128 ], num_decoder_layers=3, aux_loss=aux_loss)


        else:
            print("backbone format err")
        self.multi_scale = multi_scale
    def forward(self, x, targets=None):
        # if self.multi_scale and self.training:
        #     sz = np.random.choice(self.multi_scale)
        #     x = F.interpolate(x, size=[sz, sz])
        x      = self.backbone(x)
        x      = self.encode(x)
        x      = self.transformer(x, targets)
        return x



    @unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
    #             m.eval()

