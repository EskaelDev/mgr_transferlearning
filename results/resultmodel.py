from trainstats import TrainStats
import sys
sys.path.insert(1, '../')


class ResultModel:
    results = []  # :TrainStats

    def __init__(self, name):
        self.name = name


class Properties:
    accuracy = 0
    k1 = 0
    k5 = 0
    model_name = ''
    train_loss_array = []
    valid_loss_array = []
    train_accuracy_array = []
    valid_accuracy_array = []
    best_epoch = 0
    total_time = 0.0
    train_time_sum = 0.0
    eval_time_sum = 0.0
