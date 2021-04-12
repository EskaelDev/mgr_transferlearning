import torch.nn as nn
import torchvision.models as models
from enum import Enum
import torch.optim as optim
from cifar10_models.densenet import densenet121 as cifar_densenet121, densenet169 as cifar_densenet169, densenet161 as cifar_densenet161
from cifar10_models.googlenet import googlenet as cifar_googlenet
from cifar10_models.vgg import vgg11_bn as cifar_vgg11_bn, vgg13_bn as cifar_vgg13_bn, vgg16_bn as cifar_vgg16_bn, vgg19_bn as cifar_vgg19_bn
from cifar10_models.resnet import resnet18 as cifar_resnet18, resnet34 as cifar_resnet34, resnet50 as cifar_resnet50
from cifar10_models.resnet_orig import resnet_orig as cifar_resnet_orig
from cifar10_models.mobilenet_v2 import mobilenet_v2 as cifar_mobilenet_v2
from cifar10_models.inception_v3 import inception_v3 as cifar_inception_v3


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class TrainedModels(AutoName):
    '''
    bn - bach normalization \n
https://pytorch.org/vision/stable/models.html#torchvision.models.alexnet \n
__possible values__
alexnet \n
vgg11 \nvgg11_bn \nvgg13 \nvgg13_bn \nvgg16 \nvgg16_bn \nvgg19 \nvgg19_bn \n
resnet18 \nresnet34 \nresnet50 \nresnet152 \n
squeezenet1_0 \nsqueezenet1_1 \n
densenet121 \ndensenet169 \ndensenet161 \ndensenet201 \n
inception_v3 \n
googlenet \n
shufflenet_v2_x0_5 \nshufflenet_v2_x1_0 \nshufflenet_v2_x1_5 \nshufflenet_v2_x2_0 \n
mobilenet_v2 \nmobilenet_v3_large \nmobilenet_v3_small \n
resnext50_32x4d \nresnext101_32x8d \n
wide_resnet50_2 \nwide_resnet101_2 \n
mnasnet0_5 \nmnasnet0_75 \nmnasnet1_0 \nmnasnet1_3 \n
cifar_densenet121 \ncifar_densenet169 \ncifar_densenet161 \n
cifar_googlenet \n
cifar_vgg11_bn \ncifar_vgg13_bn \ncifar_vgg16_bn \ncifar_vgg19_bn \n
cifar_resnet18 \ncifar_resnet34 \ncifar_resnet50 \n
cifar_resnet_orig \n
cifar_mobilenet_v2 \n
cifar_inception_v3 \n

    '''
# region pytorch_models
    alexnet = auto()
# region vgg
    vgg11 = auto()
    vgg11_bn = auto()
    vgg13 = auto()
    vgg13_bn = auto()
    vgg16 = auto()
    vgg16_bn = auto()
    vgg19 = auto()
    vgg19_bn = auto()
# endregion
# region resnet
    resnet18 = auto()
    resnet34 = auto()
    resnet50 = auto()
    resnet152 = auto()
# endregion
# region squezenet
    squeezenet1_0 = auto()
    squeezenet1_1 = auto()
# endregion
# region densenet
    densenet121 = auto()
    densenet169 = auto()
    densenet161 = auto()
    densenet201 = auto()
# endregion
    inception_v3 = auto()
    googlenet = auto()
# region shufflenet
    shufflenet_v2_x0_5 = auto()
    shufflenet_v2_x1_0 = auto()
    shufflenet_v2_x1_5 = auto()
    shufflenet_v2_x2_0 = auto()
# endregion
# region mobilenet
    mobilenet_v2 = auto()
    mobilenet_v3_large = auto()
    mobilenet_v3_small = auto()
# endregion
# region resnext
    resnext50_32x4d = auto()
    resnext101_32x8d = auto()
# endregion
# region wide_resnet
    wide_resnet50_2 = auto()
    wide_resnet101_2 = auto()
# endregion
# region mnasnet
    mnasnet0_5 = auto()
    mnasnet0_75 = auto()
    mnasnet1_0 = auto()
    mnasnet1_3 = auto()
# endregion

# endregion
# region cifar_models
    cifar_densenet121 = auto()
    cifar_densenet169 = auto()
    cifar_densenet161 = auto()
    cifar_googlenet = auto()
    cifar_vgg11_bn = auto()
    cifar_vgg13_bn = auto()
    cifar_vgg16_bn = auto()
    cifar_vgg19_bn = auto()
    cifar_resnet18 = auto()
    cifar_resnet34 = auto()
    cifar_resnet50 = auto()
    cifar_resnet_orig = auto()
    cifar_mobilenet_v2 = auto()
    cifar_inception_v3 = auto()
# endregion


def get_model(model: TrainedModels, class_num: int, train_on_gpu=False):

    selected_model = None
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
# region ifs
    if model == TrainedModels.alexnet:
        selected_model, input_size = get_alexnet(class_num)
# region vgg
    elif model == TrainedModels.vgg11:
        selected_model, input_size = get_vgg11(class_num)
    elif model == TrainedModels.vgg11_bn:
        selected_model, input_size = get_vgg11_bn(class_num)
    elif model == TrainedModels.vgg13:
        selected_model, input_size = get_vgg13(class_num)
    elif model == TrainedModels.vgg13_bn:
        selected_model, input_size = get_vgg13_bn(class_num)
    elif model == TrainedModels.vgg16:
        selected_model, input_size = get_vgg16(class_num)
    elif model == TrainedModels.vgg16_bn:
        selected_model, input_size = get_vgg16_bn(class_num)
    elif model == TrainedModels.vgg19:
        selected_model, input_size = get_vgg19(class_num)
    elif model == TrainedModels.vgg19_bn:
        selected_model, input_size = get_vgg19_bn(class_num)
# endregion
# region resnet
    elif model == TrainedModels.resnet18:
        selected_model, input_size = get_resnet18(class_num)
    elif model == TrainedModels.resnet34:
        selected_model, input_size = get_resnet34(class_num)
    elif model == TrainedModels.resnet50:
        selected_model, input_size = get_resnet50(class_num)
    elif model == TrainedModels.resnet152:
        selected_model, input_size = get_resnet152(class_num)
# endregion
# region squezenet
    elif model == TrainedModels.squeezenet1_0:
        selected_model, input_size = get_squeezenet1_0(class_num)
    elif model == TrainedModels.squeezenet1_1:
        selected_model, input_size = get_squeezenet1_1(class_num)
# endregion
# region densenet
    elif model == TrainedModels.densenet121:
        selected_model, input_size = get_densenet121(class_num)
    elif model == TrainedModels.densenet169:
        selected_model, input_size = get_densenet169(class_num)
    elif model == TrainedModels.densenet161:
        selected_model, input_size = get_densenet161(class_num)
    elif model == TrainedModels.densenet201:
        selected_model, input_size = get_densenet201(class_num)
# endregion
    elif model == TrainedModels.inception_v3:
        selected_model, input_size = get_inception_v3(class_num)
    elif model == TrainedModels.googlenet:
        selected_model, input_size = get_googlenet(class_num)
# region shuflenet
    elif model == TrainedModels.shufflenet_v2_x0_5:
        selected_model, input_size = get_shufflenet_v2_x0_5(class_num)
    elif model == TrainedModels.shufflenet_v2_x1_0:
        selected_model, input_size = get_shufflenet_v2_x1_0(class_num)
    elif model == TrainedModels.shufflenet_v2_x1_5:
        selected_model, input_size = get_shufflenet_v2_x1_5(class_num)
    elif model == TrainedModels.shufflenet_v2_x2_0:
        selected_model, input_size = get_shufflenet_v2_x2_0(class_num)
# endregion
# region mobilenet
    elif model == TrainedModels.mobilenet_v2:
        selected_model, input_size = get_mobilenet_v2(class_num)
    elif model == TrainedModels.mobilenet_v3_large:
        selected_model, input_size = get_mobilenet_v3_large(class_num)
    elif model == TrainedModels.mobilenet_v3_small:
        selected_model, input_size = get_mobilenet_v3_small(class_num)
# endregion
# region resnext
    elif model == TrainedModels.resnext50_32x4d:
        selected_model, input_size = get_resnext50_32x4d(class_num)
    elif model == TrainedModels.resnext101_32x8d:
        selected_model, input_size = get_resnext101_32x8d(class_num)
# endregion
# region wide resnet
    elif model == TrainedModels.wide_resnet50_2:
        selected_model, input_size = get_wide_resnet50_2(class_num)
    elif model == TrainedModels.wide_resnet101_2:
        selected_model, input_size = get_wide_resnet101_2(class_num)
# endregion
# region mnasnet
    elif model == TrainedModels.mnasnet0_5:
        selected_model, input_size = get_mnasnet0_5(class_num)
    elif model == TrainedModels.mnasnet0_75:
        selected_model, input_size = get_mnasnet0_75(class_num)
    elif model == TrainedModels.mnasnet1_0:
        selected_model, input_size = get_mnasnet1_0(class_num)
    elif model == TrainedModels.mnasnet1_3:
        selected_model, input_size = get_mnasnet1_3(class_num)
# endregion
# endregion
# region cifar
    if selected_model is None:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]

    elif model == TrainedModels.cifar_densenet121:
        selected_model, input_size = get_cifar_densenet121(class_num)
    elif model == TrainedModels.cifar_densenet169:
        selected_model, input_size = get_cifar_densenet169(class_num)
    elif model == TrainedModels.cifar_densenet161:
        selected_model, input_size = get_cifar_densenet161(class_num)
    elif model == TrainedModels.cifar_googlenet:
        selected_model, input_size = get_cifar_googlenet(class_num)
    elif model == TrainedModels.cifar_vgg11_bn:
        selected_model, input_size = get_cifar_vgg11_bn(class_num)
    elif model == TrainedModels.cifar_vgg13_bn:
        selected_model, input_size = get_cifar_vgg13_bn(class_num)
    elif model == TrainedModels.cifar_vgg16_bn:
        selected_model, input_size = get_cifar_vgg16_bn(class_num)
    elif model == TrainedModels.cifar_vgg19_bn:
        selected_model, input_size = get_cifar_vgg19_bn(class_num)
    elif model == TrainedModels.cifar_resnet18:
        selected_model, input_size = get_cifar_resnet18(class_num)
    elif model == TrainedModels.cifar_resnet34:
        selected_model, input_size = get_cifar_resnet34(class_num)
    elif model == TrainedModels.cifar_resnet50:
        selected_model, input_size = get_cifar_resnet50(class_num)
    elif model == TrainedModels.cifar_resnet_orig:
        selected_model, input_size = get_cifar_resnet_orig(class_num)
    elif model == TrainedModels.cifar_mobilenet_v2:
        selected_model, input_size = get_cifar_mobilenet_v2(class_num)
    elif model == TrainedModels.cifar_inception_v3:
        selected_model, input_size = get_cifar_inception_v3(class_num)
# endregion
    if train_on_gpu:
        selected_model.cuda()

    optimizer = get_optimizer(selected_model)
    return selected_model, input_size, mean, std, optimizer


# region models

def get_alexnet(class_num):
    model = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224
# region vgg


def get_vgg11(class_num):
    model = models.vgg11(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg11_bn(class_num):
    model = models.vgg11_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg13(class_num):
    model = models.vgg13(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg13_bn(class_num):
    model = models.vgg13_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg16(class_num):
    model = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg16_bn(class_num):
    model = models.vgg16_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg19(class_num):
    model = models.vgg19(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_vgg19_bn(class_num):
    model = models.vgg19_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# region resnet


def get_resnet18(class_num):
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_resnet34(class_num):
    model = models.resnet34(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_resnet50(class_num):
    model = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_resnet152(class_num):
    model = models.resnet152(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# region squezenet


def get_squeezenet1_0(class_num):
    model = models.squeezenet1_0(pretrained=True)
    set_parameter_requires_grad(model)

    model.classifier[1] = nn.Conv2d(
        512, class_num, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = class_num
    return model, 224


def get_squeezenet1_1(class_num):
    model = models.squeezenet1_1(pretrained=True)
    set_parameter_requires_grad(model)

    model.classifier[1] = nn.Conv2d(
        512, class_num, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = class_num
    return model, 224
# endregion
# region densenet


def get_densenet121(class_num):
    model = models.densenet121(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_densenet169(class_num):
    model = models.densenet169(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_densenet161(class_num):
    model = models.densenet161(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_densenet201(class_num):
    model = models.densenet201(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion


def get_inception_v3(class_num):
    model = models.inception_v3(pretrained=True, aux_logits=False)
    set_parameter_requires_grad(model)

    # n_inputs = model.AuxLogits.fc.in_features
    # model.AuxLogits.fc = nn.Linear(n_inputs, class_num)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 299


def get_googlenet(class_num):
    model = models.googlenet(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224

# region shufflenet


def get_shufflenet_v2_x0_5(class_num):
    model = models.shufflenet_v2_x0_5(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 244


def get_shufflenet_v2_x1_0(class_num):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 244


def get_shufflenet_v2_x1_5(class_num):
    model = models.shufflenet_v2_x1_5(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 244


def get_shufflenet_v2_x2_0(class_num):
    model = models.shufflenet_v2_x2_0(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 244
# endregion
# region mobilenet


def get_mobilenet_v2(class_num):
    model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 244


def get_mobilenet_v3_large(class_num):
    model = models.mobilenet_v3_large(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_mobilenet_v3_small(class_num):
    model = models.mobilenet_v3_small(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# region resnext


def get_resnext50_32x4d(class_num):
    model = models.resnext50_32x4d(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_resnext101_32x8d(class_num):
    model = models.resnext50_32x4d(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# region wide resnet


def get_wide_resnet50_2(class_num):
    model = models.wide_resnet50_2(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_wide_resnet101_2(class_num):
    model = models.wide_resnet101_2(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# region mnasnet


def get_mnasnet0_5(class_num):
    model = models.mnasnet0_5(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_mnasnet0_75(class_num):
    model = models.mnasnet0_75(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_mnasnet1_0(class_num):
    model = models.mnasnet1_0(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_mnasnet1_3(class_num):
    model = models.mnasnet1_3(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 224
# endregion
# endregion

# region cifar


def get_cifar_densenet121(class_num):
    model = cifar_densenet121(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_densenet169(class_num):
    model = cifar_densenet169(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_densenet161(class_num):
    model = cifar_densenet161(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_googlenet(class_num):
    model = cifar_googlenet(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_vgg11_bn(class_num):
    model = cifar_vgg11_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_vgg13_bn(class_num):
    model = cifar_vgg13_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_vgg16_bn(class_num):
    model = cifar_vgg16_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_vgg19_bn(class_num):
    model = cifar_vgg19_bn(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_resnet18(class_num):
    model = cifar_resnet18(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_resnet34(class_num):
    model = cifar_resnet34(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_resnet50(class_num):
    model = cifar_resnet50(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_resnet_orig(class_num):
    model = cifar_resnet_orig(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_cifar_mobilenet_v2(class_num):
    model = cifar_mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, class_num)

    return model, 244


def get_cifar_inception_v3(class_num):
    model = cifar_inception_v3(pretrained=True)
    set_parameter_requires_grad(model)

    # n_inputs = model.AuxLogits.fc.in_features
    # model.AuxLogits.fc = nn.Linear(n_inputs, class_num)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 299
# endregion


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_optimizer(model):
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print("\t", name)
    optimizer = optim.Adam(params_to_update, lr=0.001)
    return optimizer
