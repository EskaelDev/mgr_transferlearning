# %%
from datetime import time
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
import pandas as pd
import pprint

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
    
def customDecoder(objDict):
    return namedtuple('X', objDict.keys())(*objDict.values())
    
def plot_array(plot_name: str, array: set, best_epoch: int):
    plt.plot(array[:best_epoch])
    plt.ylabel(plot_name)
    plt.show()

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
m_arrays = ['TrainAccuracy',
'TrainLoss',
'ValidAccuracy',
'ValidLoss',]
# %%  

with open("averageResults.json") as f:
    averageResults = json.load(f)    

# %%  
def PlotResultArrays(results, index:int, arrayType:str, legendName:str):
    minArr = np.array(results[index][f"Min{arrayType}"])
    maxArr =  np.array(results[index][f"Max{arrayType}"])
    avgArr =  np.array(results[index][f"Avg{arrayType}"])
    length = list(range(0, len(results[index][f"Min{arrayType}"])))
    modelName = results[index]['ModelName']

    maxColor = "#b41f44"
    minColor = "#1F77B4"
    avgColor = "#000"

    fig, ax = plt.subplots()

    ax.set_title(modelName + ' '+ arrayType)
    ax.fill_between(length, minArr, maxArr, alpha=0.3)
    
    # ~ do odkomentowania jeżeli polski
    # ax.plot(length, avgArr, color=avgColor, linewidth=0.5, label=f'Średnie wartość {legendName}')
    # ax.plot(length, minArr, color=minColor, linewidth=0.3, label=f'Najniższe wartości {legendName}')
    # ax.plot(length, maxArr,color=maxColor, linewidth=0.3, label=f'Najwyższe wartości {legendName}')

    ax.plot(length, avgArr, color=avgColor, linewidth=0.5, label=f'Avg')
    ax.plot(length, minArr, color=minColor, linewidth=0.3, label=f'Min')
    ax.plot(length, maxArr,color=maxColor, linewidth=0.3, label=f'Max')

    ax.axvline(x=results[index]["AvgBestEpoch"], ymin=0, ymax=1, color=avgColor, alpha=0.3)
    ax.axvline(x=results[index]["MaxBestEpoch"], ymin=0, ymax=1, color=maxColor, alpha=0.3)
    ax.axvline(x=results[index]["MinBestEpoch"], ymin=0, ymax=1, color=minColor, alpha=0.3)
    # ~ do odkomentowania jeżeli polski
    # ax.set_xlabel(f'Najniższa epoka nauczenia: {results[index]["MinBestEpoch"]} Średnia epoka nauczenia: {results[index]["AvgBestEpoch"]}\nNajwyższa epoka nauczenia: {results[index]["MaxBestEpoch"]} ')
    ax.set_xlabel(f'Min epoch: {results[index]["MinBestEpoch"]}; Avg epoch: {results[index]["AvgBestEpoch"]}; Max epoch: {results[index]["MaxBestEpoch"]} ')
    

    ax.legend()
    fig.tight_layout()
# %%  
for i in range(0, len(averageResults)):
    for arrayType in m_arrays:
        PlotResultArrays(averageResults, i, arrayType, 'fukncji błędu')
    
    if i == 3:
        break

# %%
# def SortByMetric(results, metric:str):
#     metricDict = []  
#     for i in range(0, len(averageResults)):
#         metricDict.append([[averageResults[i]['ModelName']], averageResults[i]['Avg'+metric]])
    
#     return metricDict
#     # return dict(sorted(metricDict.items(), key=lambda item: item[1]))
    

# # pprint.pprint(dictionary)
# print(SortByMetric(averageResults, 'TotalTime'))
# print()
# print(SortByMetric(averageResults, 'Accuracy'))
# %%
metricDict = []  
for i in range(0, len(averageResults)):
    metricDict.append([averageResults[i]['ModelName'], averageResults[i]['AvgTotalTime'],averageResults[i]['AvgAccuracy']])

metricDict
# %%

labels = ['cifar_densenet121',
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
'vgg19_bn',]
timeNormalized = [0.297524952,
0.653659876,
0.288108779,
1,
0.258052831,
0.091540821,
0.060416227,
0.226203424,
0.160706849,
0.090673421,
0.125951657,
0.173946699,
0.265451556,
0.570352765,
0.146352367,
0.125382537,
0.230485928,
0.099670543,
0.073030908,
0.084089612,
0.130317146,
0.141321976,
0.125653598,
0.112975012,
0.104277939,]
accuracyNormalized = [0.350694458,
0.328125,
0.348958333,
0.305555563,
0.342013875,
0.434027792,
0.369791667,
0.364583333,
0.315972229,
0.321180563,
0.258680563,
0.192708333,
0.96875,
1,
0.991319417,
0.954861083,
0.958333333,
0.96875,
0.953125,
0.9375,
0.96875,
0.953125,
0.960069417,
0.928819417,
0.923611083,]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 5), dpi=80)

rects1 = ax.bar(x - width/2, timeNormalized, width, label='Czas znormalizowany')
rects2 = ax.bar(x + width/2, accuracyNormalized, width, label='Celność znormalizowana')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
plt.xticks(rotation=45)
plt.grid(True, axis='y', label=labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()