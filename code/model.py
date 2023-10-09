import sys

from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

encoders = {
    "resnet34": {
        "encoder": models.resnet34,
        "out_shape": 512
    },
    "resnet50": {
        "encoder": models.resnet50,
        "out_shape": 2048
    },
    "resnet50_cbam": {
        "encoder": models.resnet50,
        "layer_shapes": [2048, 1024, 512, 256, 64],
        "out_shape": 2048
    }
    }
class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction=4,attention_kernel_size=3,position_encode=False):
        super(CBAM_Module, self).__init__()
        self.position_encode=position_encode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        if self.position_encode:
            k=3
        else:
            k=2
        self.conv_after_concat = nn.Conv2d(k, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()
        self.position_encoded=None

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        if self.position_encode:
            if self.position_encoded is None:

                pos_enc=get_sinusoid_encoding_table(h,w)
                pos_enc=Variable(torch.FloatTensor(pos_enc),requires_grad=False)
                if x.is_cuda:
                    pos_enc=pos_enc.cuda()
                self.position_encoded=pos_enc
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        if self.position_encode:
            pos_enc=self.position_encoded
            pos_enc = pos_enc.view(1, 1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat((avg, mx,pos_enc), 1)
        else:
            x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x
    
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CnnModel(nn.Module):

    def __init__(self, num_classes, encoder="resnet50", pretrained=False, pool_type="avg"):
        super().__init__()
        self.net = encoders[encoder]["encoder"](pretrained=pretrained)
        if encoder == "resnet50_cbam":
            self.net.layer1 = nn.Sequential(self.net.layer1,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][3]))
            self.net.layer2 = nn.Sequential(self.net.layer2,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][2]))
            self.net.layer3 = nn.Sequential(self.net.layer3,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][1]))
            self.net.layer4 = nn.Sequential(self.net.layer4,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][0]))

        if encoder in ["resnet34", "resnet50", "resnet50_cbam"]:
            if pool_type == "concat":
                self.net.avgpool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.fc = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        elif encoder == "inceptionresnetv2":
            if pool_type == "concat":
                self.net.avgpool_1a = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool_1a = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        else:
            if pool_type == "concat":
                self.net.avg_pool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avg_pool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class EnsembleModel(nn.Module):
    def __init__(self, num_classes=1, num_models=None):
        super(EnsembleModel, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.models = nn.ModuleList([CnnModel(num_classes=1)])
    def forward(self, x):
        # Forward pass through each model in the ensemble
        outputs = [model(x[i]) for i, model in enumerate(self.models)]
        return outputs
