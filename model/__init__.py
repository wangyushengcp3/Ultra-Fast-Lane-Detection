import copy
from .mobilenetv2 import MobileNetV2
from .resnet import resnet

def build_backbone(name):
    if name == 'resnet':
        layer = name.split('_')[1]
        return resnet(layer)
    elif name == 'mobilenetv2':
        return MobileNetV2()
    else:
        raise NotImplementedError
    # elif name == 'ShuffleNetV2':
    #     return ShuffleNetV2(**backbone_cfg)
    # elif name == 'GhostNet':
    #     return GhostNet(**backbone_cfg)
        
    
    # elif name == 'EfficientNetLite':
    #     return EfficientNetLite(**backbone_cfg)
    # elif name == 'CustomCspNet':
    #     return CustomCspNet(**backbone_cfg)
    # # elif name == 'RepVGG':
    # #     return RepVGG(**backbone_cfg)


