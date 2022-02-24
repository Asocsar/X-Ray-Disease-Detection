import torch.nn as nn
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import os

#%%
class DenseNet121_Pretrained(nn.Module):
    def __init__(self, out_size, weights):
        super(DenseNet121_Pretrained, self).__init__()
        pretrained_weights = torch.load(weights)
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        self.densenet121.load_state_dict(pretrained_weights, strict=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

#%%
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

#%%
class DenseNet121_Mixed(nn.Module):
    def __init__(self, out_size, weights):
        super(DenseNet121_Mixed, self).__init__()
        pretrained_weights = torch.load(weights)
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        self.densenet121.load_state_dict(pretrained_weights, strict=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
#%%
class Efficient(nn.Module):
    def __init__(self, Name, path_weights, N_classes):
        super(Efficient, self).__init__()

        self.efficient = EfficientNet.from_name(Name)
        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, N_classes)

        pretrained_weights = torch.load(os.path.join(path_weights, Name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.efficient(x)
        x = self.sigmoid(x)
        return x


#%%
class Efficient_Mixed(nn.Module):
    def __init__(self, Name, path_weights, N_classes):
        super(Efficient_Mixed, self).__init__()

        self.efficient = EfficientNet.from_name(Name)
        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, N_classes)

        pretrained_weights = torch.load(os.path.join(path_weights, Name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)


    def forward(self, x):
        x = self.efficient(x)
        return x

class Efficient_NoPretrain(nn.Module):
    def __init__(self, Name, N_classes):
        super(Efficient_NoPretrain, self).__init__()
        self.efficient = EfficientNet.from_name(Name)
        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, N_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.efficient(x)
        x = self.sigmoid(x)
        return x