import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BDD100kModel(nn.Module):
    def __init__(self, num_classes, backbone:nn.Module, size=(776,776)):
        super(BDD100kModel, self).__init__()
        
        # if backbone == None:
        #   # Load ResNet-50-Dilated as backbone
        #   self.backbone = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
          
        #   # Replace the last layer of the backbone to output the desired number of classes
        #   self.backbone.classifier = nn.Conv2d(2048, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # else:
        self.backbone = backbone

        # Upsampling layer to increase the resolution of the output
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Pass input through backbone
        # x = self.backbone(x)['out']
        x = self.backbone(x)
        # Upsample to increase resolution
        x = self.upsample(x)
        
        return x