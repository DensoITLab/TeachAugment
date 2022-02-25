from . import resnet
from . import pyramidnet
from . import wide_resnet
from .shakeshake import shake_resnet


def build_model(model_name, num_classes=10):
    if model_name in ['wideresnet-28-10', 'wrn-28-10']:
        model = wide_resnet.WideResNet(28, 10, 0, num_classes)

    elif model_name in ['wideresnet-40-2', 'wrn-40-2']:
        model = wide_resnet.WideResNet(40, 2, 0, num_classes)

    elif model_name in ['shakeshake26_2x32d', 'ss32']:
        model = shake_resnet.ShakeResNet(26, 32, num_classes)

    elif model_name in ['shakeshake26_2x96d', 'ss96']:
        model = shake_resnet.ShakeResNet(26, 96, num_classes)

    elif model_name in ['shakeshake26_2x112d', 'ss112']:
        model = shake_resnet.ShakeResNet(26, 112, num_classes)

    elif model_name == 'pyramidnet':
        model = pyramidnet.PyramidNet('cifar10', depth=272, alpha=200, num_classes=num_classes, bottleneck=True)

    elif model_name == 'resnet200':
        model = resnet.ResNet('imagenet', 200, num_classes, True)

    elif model_name == 'resnet50':
        model = resnet.ResNet('imagenet', 50, num_classes, True)

    return model
