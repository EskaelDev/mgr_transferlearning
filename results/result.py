# %%
import sys
sys.path.insert(1, '../')
import glob
import numpy as np
import matplotlib.pyplot as plt
from json import JSONEncoder
from collections import namedtuple
import json
from resultmodel import ResultModel, Properties
from datasetmodel import uc_landuse_ds
from datasetmodel import DatasetModel
from trainstats import TrainStats
# %%
# insert at 1, 0 is the script path (or '' in REPL)
# %%
# %%


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# %%


def customDecoder(objDict):
    return namedtuple('X', objDict.keys())(*objDict.values())


def plot_array(plot_name: str, array: set, best_epoch: int):
    plt.plot(array[:best_epoch])
    plt.ylabel(plot_name)
    plt.show()


# %%
m_names = ['cifar_densenet121',
           'cifar_densenet161',
           'cifar_densenet169',
           'cifar_googlenet',
           'cifar_mobilenet_v2',
           'cifar_resnet18',
           'cifar_resnet34',
           'cifar_resnet50',
           'cifar_vgg11_bn',
           'cifar_vgg13_bn',
           'cifar_vgg16_bn',
           'cifar_vgg19_bn',
           'densenet121',
           'densenet161',
           'densenet169',
           'googlenet',
           'inception_v3',
           'mobilenet_v2',
           'resnet18',
           'resnet34',
           'resnet50',
           'vgg11_bn',
           'vgg13_bn',
           'vgg16_bn',
           'vgg19_bn']
all_results_array = {}
for name in m_names:
    all_results_array[name] = ResultModel(name)

# %%
# %%
working_ds = uc_landuse_ds
working_ds.name
result_files = (glob.glob(f"data/{working_ds.name}/*.json"))

# %%

for pth in result_files:
    with open(pth) as f:
        try:
            current_file: TrainStats = json.load(f, object_hook=customDecoder)
        except Exception:
            print(
                f"{bcolors.FAIL}Error in file: {bcolors.WARNING}{pth}{bcolors.ENDC}")
    
    all_results_array[current_file.model_name].results.append(current_file)
    # for item in all_results_array:
    #     if current_file.model_name == item.name:
    #         item.results.append(current_file)
# %%
best_epoch_array = {}
total_time_array = {}
train_time_array = {}
eval_time_array = {}
accuracy_array = {}
# for name in m_names:
#     best_epoch_array[name] = []
#     total_time_array[name] = []
#     train_time_array[name] = []
#     eval_time_array[name] = []
#     accuracy_array[name] = []


for one_file in all_results_array:
    if one_file.name not in best_epoch_array:
        best_epoch_array[one_file.name] = []
        total_time_array[one_file.name] = []
        train_time_array[one_file.name] = []
        eval_time_array[one_file.name] = []
        accuracy_array[one_file.name] = []
    for one_file_result in one_file.results:
        best_epoch_array[one_file.name].append(one_file_result.best_epoch)
        total_time_array[one_file.name].append(one_file_result.total_time)
        train_time_array[one_file.name].append(one_file_result.train_time_sum)
        eval_time_array[one_file.name].append(one_file_result.eval_time_sum)
        accuracy_array[one_file.name].append(one_file_result.accuracy)

print(best_epoch_array["googlenet"])
