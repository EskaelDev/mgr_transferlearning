from pathlib import Path
from datetime import datetime
from datasetmodel import DatasetModel
import numpy as np
import json


class TrainStats:
    accuracy = 0
    k1 = 0
    k5 = 0
    class_accuracy = {}
    confusion = np.ndarray(shape=(2, 2))

    def __init__(self, model_name, train_loss_array, valid_loss_array, train_accuracy_array, valid_accuracy_array, best_epoch, total_time, train_time_sum, eval_time_sum):
        self.model_name = model_name
        self.train_loss_array = train_loss_array
        self.valid_loss_array = valid_loss_array
        self.train_accuracy_array = train_accuracy_array
        self.valid_accuracy_array = valid_accuracy_array
        self.best_epoch = best_epoch
        self.total_time = total_time
        self.train_time_sum = train_time_sum
        self.eval_time_sum = eval_time_sum

    def save(self, working_ds: DatasetModel):
        now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        dir_path = f'drive/MyDrive/results/{working_ds.name}_results/'
        file_name = f'{self.model_name}_{now}'

        Path(dir_path).mkdir(parents=True, exist_ok=True)

        with open(f'{dir_path}{file_name}.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)
        print(f'💾File {file_name}.json saved at {dir_path}')
