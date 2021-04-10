import torch.nn as nn
import torchvision.models as models
from enum import Enum


class TrainedModels(Enum):
    resnet18 = 1
    alexnet = 2
    squeezenet1_0 = 3
    vgg16 = 4
    densenet161 = 5
    inception_v3 = 6
    googlenet = 7
    shufflenet_v2_x1_0 = 8
    mobilenet_v2 = 9
    mobilenet_v3_large = 10
    mobilenet_v3_small = 11
    resnext50_32x4d = 12
    wide_resnet50_2 = 13
    mnasnet1_0 = 14


def get_model(model: TrainedModels, class_num: int, train_on_gpu=False):

    if model == TrainedModels.resnet18:
        return get_resnet18(train_on_gpu, class_num)

    if model == TrainedModels.alexnet:
        return get_alexnet(train_on_gpu, class_num)

    if model == TrainedModels.squeezenet1_0:
        return get_squeezenet1_0(train_on_gpu, class_num)

    if model == TrainedModels.vgg16:
        return get_vgg16(train_on_gpu, class_num)

    if model == TrainedModels.densenet161:
        return get_densenet161(train_on_gpu, class_num)

    if model == TrainedModels.inception_v3:
        return get_inception_v3(train_on_gpu, class_num)

    if model == TrainedModels.googlenet:
        return get_googlenet(train_on_gpu, class_num)

    if model == TrainedModels.shufflenet_v2_x1_0:
        return get_shufflenet_v2_x1_0(train_on_gpu, class_num)

    if model == TrainedModels.mobilenet_v2:
        return get_mobilenet_v2(train_on_gpu, class_num)

    if model == TrainedModels.mobilenet_v3_large:
        return get_mobilenet_v3_large(train_on_gpu, class_num)

    if model == TrainedModels.mobilenet_v3_small:
        return get_mobilenet_v3_small(train_on_gpu, class_num)

    if model == TrainedModels.resnext50_32x4d:
        return get_resnext50_32x4d(train_on_gpu, class_num)

    if model == TrainedModels.wide_resnet50_2:
        return get_wide_resnet50_2(train_on_gpu, class_num)

    if model == TrainedModels.mnasnet1_0:
        return get_mnasnet1_0(train_on_gpu, class_num)


def get_resnet18(train_on_gpu, class_num):
    model = models.resnet18(pretrained=True)
    return model


def get_alexnet(train_on_gpu, class_num):
    model = models.alexnet(pretrained=True)
    return model


def get_squeezenet1_0(train_on_gpu, class_num):
    model = models.squeezenet1_0(pretrained=True)
    return model


def get_vgg16(train_on_gpu, class_num):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters(train_on_gpu, class_num):
        param.requires_grad = False

    n_inputs = model.classifier[6].in_features

    # add last linear layer (n_inputs -> 5 flower classes)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, class_num)

    model.classifier[6] = last_layer

    # if GPU is available, move the model to GPU
    if train_on_gpu:
        model.cuda()
    return model


def get_densenet161(train_on_gpu, class_num):
    model = models.densenet161(pretrained=True)
    return model


def get_inception_v3(train_on_gpu, class_num):
    model = models.inception_v3(pretrained=True)
    return model


def get_googlenet(train_on_gpu, class_num):
    model = models.googlenet(pretrained=True)
    return model


def get_shufflenet_v2_x1_0(train_on_gpu, class_num):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    return model


def get_mobilenet_v2(train_on_gpu, class_num):
    model = models.mobilenet_v2(pretrained=True)
    return model


def get_mobilenet_v3_large(train_on_gpu, class_num):
    model = models.mobilenet_v3_large(pretrained=True)
    return model


def get_mobilenet_v3_small(train_on_gpu, class_num):
    model = models.mobilenet_v3_small(pretrained=True)
    return model


def get_resnext50_32x4d(train_on_gpu, class_num):
    model = models.resnext50_32x4d(pretrained=True)
    return model


def get_wide_resnet50_2(train_on_gpu, class_num):
    model = models.wide_resnet50_2(pretrained=True)
    return model


def get_mnasnet1_0(train_on_gpu, class_num):
    model = models.mnasnet1_0(pretrained=True)
    return model
