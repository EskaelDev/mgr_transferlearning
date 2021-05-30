# %%
import sys
sys.path.insert(1, '../')

import seaborn as sn
import pprint
import pandas as pd
from trainstats import TrainStats
from datasetmodel import DatasetModel
from datasetmodel import uc_landuse_ds, resisc_ds
from resultmodel import ResultModel, Properties
import json
from collections import namedtuple
from json import JSONEncoder
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import time


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


# m_names = ['cifar_densenet121',
#            'cifar_densenet161',
#            'cifar_densenet169',
#            'cifar_googlenet',
#            'cifar_mobilenet_v2',
#            'cifar_resnet18',
#            'cifar_resnet34',
#            'cifar_resnet50',
#            'cifar_vgg11_bn',
#            'cifar_vgg13_bn',
#            'cifar_vgg16_bn',
#            'cifar_vgg19_bn',
#            'densenet121',
#            'densenet161',
#            'densenet169',
#            'googlenet',
#            'inception_v3',
#            'mobilenet_v2',
#            'resnet18',
#            'resnet34',
#            'resnet50',
#            'vgg11_bn',
#            'vgg13_bn',
#            'vgg16_bn',
#            'vgg19_bn']
m_names = ['densenet169',
           'mobilenet_v2',
           'resnet18',
           'vgg13_bn']           
m_arrays = ['TrainAccuracy',
            'TrainLoss',
            'ValidAccuracy',
            'ValidLoss', ]
working_ds = resisc_ds
# %%

with open("averageResiscResults.json") as f:
    averageResults = json.load(f)

# %%


def PlotResultArrays(results, index: int, arrayType: str, legendName: str):
    minArr = np.array(results[index][f"Min{arrayType}"])
    maxArr = np.array(results[index][f"Max{arrayType}"])
    avgArr = np.array(results[index][f"Avg{arrayType}"])
    length = list(range(0, len(results[index][f"Min{arrayType}"])))
    modelName = results[index]['ModelName']

    maxColor = "#b41f44"
    minColor = "#1F77B4"
    avgColor = "#000"

    fig, ax = plt.subplots()

    ax.set_title(modelName + ' ' + arrayType)
    ax.fill_between(length, minArr, maxArr, alpha=0.3)

    # ~ do odkomentowania jeżeli polski
    # ax.plot(length, avgArr, color=avgColor, linewidth=0.5, label=f'Średnie wartość {legendName}')
    # ax.plot(length, minArr, color=minColor, linewidth=0.3, label=f'Najniższe wartości {legendName}')
    # ax.plot(length, maxArr,color=maxColor, linewidth=0.3, label=f'Najwyższe wartości {legendName}')

    ax.plot(length, avgArr, color=avgColor, linewidth=0.5, label=f'Avg')
    ax.plot(length, minArr, color=minColor, linewidth=0.3, label=f'Min')
    ax.plot(length, maxArr, color=maxColor, linewidth=0.3, label=f'Max')

    ax.axvline(x=results[index]["AvgBestEpoch"], ymin=0,
               ymax=1, color=avgColor, alpha=0.3)
    ax.axvline(x=results[index]["MaxBestEpoch"], ymin=0,
               ymax=1, color=maxColor, alpha=0.3)
    ax.axvline(x=results[index]["MinBestEpoch"], ymin=0,
               ymax=1, color=minColor, alpha=0.3)
    # ~ do odkomentowania jeżeli polski
    # ax.set_xlabel(f'Najniższa epoka nauczenia: {results[index]["MinBestEpoch"]} Średnia epoka nauczenia: {results[index]["AvgBestEpoch"]}\nNajwyższa epoka nauczenia: {results[index]["MaxBestEpoch"]} ')
    ax.set_xlabel(
        f'Min epoch: {results[index]["MinBestEpoch"]}; Avg epoch: {results[index]["AvgBestEpoch"]}; Max epoch: {results[index]["MaxBestEpoch"]} ')

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plots/resisc45/{modelName}-{arrayType}", dpi=200)
    return ax


# ! plot błedu i acc
# %%
for i in range(0, len(averageResults)):
    for arrayType in m_arrays:
        figu = PlotResultArrays(averageResults, i, arrayType, 'fukncji błędu')
        # figu.figure.savefig(f"arrayType")
    # if i == 1:
    #     break

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
    metricDict.append([averageResults[i]['ModelName'], averageResults[i]
                       ['AvgTotalTime'], averageResults[i]['AvgAccuracy']])

metricDict
# ! Time | Accuracy
# ! time i acc wziać z excela dla resisc
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
          'vgg19_bn', ]
timeNormalized = [
    0.712834059,
    0.738805215,
    0.570747986,
    1 ]
accuracyNormalized = [
    1,
    0.968920196,
    0.936602273,
    0.938088109 ]

x = np.arange(len(m_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

rects1 = ax.bar(x - width/2, timeNormalized,
                width, label='Czas znormalizowany')
rects2 = ax.bar(x + width/2, accuracyNormalized,
                width, label='Dokładność znormalizowana')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
# plt.xticks(rotation=45)
ax.set_xticklabels(m_names)
plt.grid(True, axis='y', label=m_names)
ax.legend(bbox_to_anchor=(0.4, 1.2))

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

# ! time i acc wziać z excela dla uc
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
          'vgg19_bn', ]
timeNormalized = [
    0.297524952,
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
    0.104277939, ]
accuracyNormalized = [
    0.350694458,
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
    0.923611083, ]

x = np.arange(len(m_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 5), dpi=50)

rects1 = ax.bar(x - width/2, timeNormalized,
                width, label='Czas znormalizowany')
rects2 = ax.bar(x + width/2, accuracyNormalized,
                width, label='Dokładność znormalizowana')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
plt.xticks(rotation=45)
ax.set_xticklabels(m_names)
plt.grid(True, axis='y', label=m_names)
# ax.legend(bbox_to_anchor=(0.4, 1.2))
ax.legend(loc=2)

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
# ! class acc
# %%
for i in range(0, len(averageResults)):    
    print(f"=========={averageResults[i]['ModelName']}==========")
    pprint.pprint(averageResults[i]['AvgClassAccuracy'])
# ! confusion matrix
# %%

def loop_inplace_sum(confusions):
    # assumes len(arrlist) > 0
    sum = confusions[0].copy()
    for a in confusions[1:]:
        sum += a
    return sum/len(confusions)


def get_plot_confusion(confusion_array, working_ds: DatasetModel):
    df_cm = pd.DataFrame(confusion_array, index=[i for i in working_ds.classes],
                         columns=[i for i in working_ds.classes])
    plt.figure(figsize=(15, 15), dpi=200)
    return sn.heatmap(df_cm, annot=True, cmap='BuPu', cbar=False)

# %%
for i in range(0, len(averageResults)):
    confusion_plot = get_plot_confusion(averageResults[i]['AvgConfusion'], working_ds)
    confusion_plot.figure.savefig(f"plots/resisc45/confusion/{averageResults[i]['ModelName']}-confusion.png")
    


# ! Rzeczy z jedną wartością
# %% 
for i in range(0, len(averageResults)):    
    print(f"=========={averageResults[i]['ModelName']}==========")
    print(f"AvgAccuracy :{averageResults[i]['AvgAccuracy']}")
    print(f"AvgEvalTime :{averageResults[i]['AvgEvalTime']}")
    print(f"AvgFMeasure :{averageResults[i]['AvgFMeasure']}")
    print(f"AvgK1 :{averageResults[i]['AvgK1']}")
    print(f"AvgK5 :{averageResults[i]['AvgK5']}")
    print(f"AvgPrecision :{averageResults[i]['AvgPrecision']}")
    print(f"AvgRecall :{averageResults[i]['AvgRecall']}")
    print(f"AvgTotalTime :{averageResults[i]['AvgTotalTime']}")
    print(f"AvgTrainTime :{averageResults[i]['AvgTrainTime']}")
    print(f"MaxAccuracy :{averageResults[i]['MaxAccuracy']}")
    print(f"MaxEvalTime :{averageResults[i]['MaxEvalTime']}")
    print(f"MaxFMeasure :{averageResults[i]['MaxFMeasure']}")
    print(f"MaxK1 :{averageResults[i]['MaxK1']}")
    print(f"MaxK5 :{averageResults[i]['MaxK5']}")
    print(f"MaxPrecision :{averageResults[i]['MaxPrecision']}")
    print(f"MaxRecall :{averageResults[i]['MaxRecall']}")
    print(f"MaxTotalTime :{averageResults[i]['MaxTotalTime']}")
    print(f"MaxTrainTime :{averageResults[i]['MaxTrainTime']}")
    print(f"MinAccuracy :{averageResults[i]['MinAccuracy']}")
    print(f"MinEvalTime :{averageResults[i]['MinEvalTime']}")
    print(f"MinFMeasure :{averageResults[i]['MinFMeasure']}")
    print(f"MinK1 :{averageResults[i]['MinK1']}")
    print(f"MinK5 :{averageResults[i]['MinK5']}")
    print(f"MinPrecision :{averageResults[i]['MinPrecision']}")
    print(f"MinRecall :{averageResults[i]['MinRecall']}")
    print(f"MinTotalTime :{averageResults[i]['MinTotalTime']}")
    print(f"MinTrainTime :{averageResults[i]['MinTrainTime']}")
    print(f"AvgBestEpoch :{averageResults[i]['AvgBestEpoch']}")
    print(f"MaxBestEpoch :{averageResults[i]['MaxBestEpoch']}")
    print(f"MinBestEpoch :{averageResults[i]['MinBestEpoch']}")
