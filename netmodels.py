import torch.nn as nn
import torchvision.models as models
from enum import Enum
import torch.optim as optim


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

    selected_model = None
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if model == TrainedModels.resnet18:
        selected_model, input_size = get_resnet18(class_num)

    if model == TrainedModels.alexnet:
        selected_model, input_size = get_alexnet(class_num)

    if model == TrainedModels.squeezenet1_0:
        selected_model, input_size = get_squeezenet1_0(class_num)

    if model == TrainedModels.vgg16:
        selected_model, input_size = get_vgg16(class_num)

    if model == TrainedModels.densenet161:
        selected_model, input_size = get_densenet161(class_num)

    if model == TrainedModels.inception_v3:
        selected_model, input_size = get_inception_v3(class_num)

    if model == TrainedModels.googlenet:
        selected_model, input_size = get_googlenet(class_num)

    if model == TrainedModels.shufflenet_v2_x1_0:
        selected_model, input_size = get_shufflenet_v2_x1_0(class_num)

    if model == TrainedModels.mobilenet_v2:
        selected_model, input_size = get_mobilenet_v2(class_num)

    if model == TrainedModels.mobilenet_v3_large:
        selected_model, input_size = get_mobilenet_v3_large(class_num)

    if model == TrainedModels.mobilenet_v3_small:
        selected_model, input_size = get_mobilenet_v3_small(class_num)

    if model == TrainedModels.resnext50_32x4d:
        selected_model, input_size = get_resnext50_32x4d(class_num)

    if model == TrainedModels.wide_resnet50_2:
        selected_model, input_size = get_wide_resnet50_2(class_num)

    if model == TrainedModels.mnasnet1_0:
        selected_model, input_size = get_mnasnet1_0(class_num)

    if selected_model is None:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]

    if train_on_gpu:
        selected_model.cuda()

    optimizer = get_optimizer(selected_model)
    return selected_model, input_size, mean, std, optimizer


def get_resnet18(class_num):
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_alexnet(class_num):
    model = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_squeezenet1_0(class_num):
    model = models.squeezenet1_0(pretrained=True)
    set_parameter_requires_grad(model)

    model.classifier[1] = nn.Conv2d(
        512, class_num, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = class_num
    return model, 224


def get_vgg16(class_num):
    model = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, class_num)

    return model, 224


def get_densenet161(class_num):
    model = models.densenet161(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier.in_features
    model.classifier = nn.Linear(n_inputs, num_classes)

    return model, 224


def get_inception_v3(class_num):
    model = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model)

    n_inputs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(n_inputs, num_classes)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, num_classes)

    return model, 299


def get_googlenet(class_num):
    model = models.googlenet(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_shufflenet_v2_x1_0(class_num):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 244


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

    return model, 244


def get_resnext50_32x4d(class_num):
    model = models.resnext50_32x4d(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_wide_resnet50_2(class_num):
    model = models.wide_resnet50_2(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, class_num)

    return model, 224


def get_mnasnet1_0(class_num):
    model = models.mnasnet1_0(pretrained=True)
    set_parameter_requires_grad(model)

    n_inputs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(n_inputs, class_num)

    return model, 224


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
