import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet_EX(nn.Module):
    def __init__(self, model):
        super(MobileNet_EX, self).__init__()
        self.features = nn.Sequential(
            *list(model.children())[0][:-1]
        )
        
    def forward(self, x):
        x = self.features(x)
        print(x.size())
        return x


class MobileNet_CL(nn.Module):
    def __init__(self, outchannel=1280, num_classes=1000):
        super(MobileNet_CL, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(320, outchannel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU6(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(outchannel, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        
        return x
    
    
def mobilenet_ex():
    model = models.mobilenet_v2(pretrained=True)
    return MobileNet_EX(model)

def mobilenet_cl(outchannel=1280, num_classes=1000):
    
    return MobileNet_CL(outchannel=outchannel, num_classes=num_classes)
