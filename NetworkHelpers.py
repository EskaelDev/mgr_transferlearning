import seaborn as sn
import pandas as pd
from datasetmodel import DatasetModel
from netparams import NetParams
from termcolor import colored
from trainstats import TrainStats
from typing import List
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def train_loop(netparams: NetParams, no_improvement=0) -> TrainStats:

    start_time = time.time()
    valid_loss_min = np.Inf
    train_loss_array = []
    valid_loss_array = []
    train_accuracy_array = []
    valid_accuracy_array = []
    train_time_sum = 0
    eval_time_sum = 0
    best_epoch = 1

    for epoch in range(1, netparams.n_epochs + 1):

        # early stopping
        if no_improvement >= netparams.max_no_improve_epochs:
            best_epoch = epoch - netparams.max_no_improve_epochs
            break

        # keep track of training and validation loss
        train_loss = 0.0
        train_loss_tmp = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        train_time = time.time()
        train_loss, train_acc = train_model(netparams=netparams,
                                            train_loss_tmp=train_loss_tmp,
                                            train_loss=train_loss,
                                            epoch=epoch)

        train_end_time = time.time() - train_time
        print(f"⌛Training epoch {epoch} took {train_end_time:.2f} seconds\n")
        train_accuracy_array.append(train_acc)

        train_time_sum += train_end_time

        ######################
        # evaluate the model #
        ######################
        eval_time = time.time()
        valid_loss_min, no_improvement, valid_acc = evaluate_model(netparams=netparams,
                                                                   train_loss=train_loss,
                                                                   valid_loss=valid_loss,
                                                                   valid_loss_min=valid_loss_min,
                                                                   valid_loss_array=valid_loss_array,
                                                                   train_loss_array=train_loss_array,
                                                                   epoch=epoch,
                                                                   no_improvement=no_improvement)

        eval_end_time = time.time() - eval_time
        print(
            f"⌛Evaluating epoch {epoch} took {(eval_end_time):.2f} seconds\n")
        eval_time_sum += eval_end_time

        valid_accuracy_array.append(valid_acc)

        if no_improvement <= 0:
            best_epoch = epoch
            print(colored('Last improvement', 'blue'))
            print(colored(f'Training took: {train_time_sum:.2f}', 'blue'))
            print(colored(f'Evaluation took: {train_time_sum:.2f}\n', 'blue'))

    total_time = time.time() - start_time
    print(
        f"🎓Total learning took {math.floor(total_time/60):02.0f}:{(total_time%60):.0f}")
    print(
        f"🏋️‍♂️Training took       {(train_time_sum/60):02.0f}:{(train_time_sum%60):02.3f}")
    print(
        f"📑Evaluation took     {(eval_time_sum/60):02.0f}:{(eval_time_sum%60):02.3f}")
    results = TrainStats(netparams.model.name, train_loss_array, valid_loss_array, train_accuracy_array,
                         valid_accuracy_array, best_epoch, total_time, train_time_sum, eval_time_sum)
    return results


def train_model(netparams: NetParams,
                train_loss_tmp,
                train_loss,
                epoch):
    netparams.model.train()
    print("Training")
    correct_outputs = 0
    for batch_i, (data, target) in enumerate(netparams.train_loader):
        # move tensors to GPU if CUDA is available
        if netparams.train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        netparams.optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = netparams.model(data)
        # calculate the batch loss
        loss = netparams.criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        netparams.optimizer.step()
        # update training loss
        train_loss_tmp += loss.item()
        train_loss += loss.item() * data.size(0)

        _, preds = torch.max(output, 1)
        correct_outputs += torch.sum(preds == target.data)

        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print(
                f'Epoch {epoch}, Batch {batch_i + 1} loss: {(train_loss_tmp / 20):.6f}')
            train_loss_tmp = 0.0

    epoch_acc = float(correct_outputs) / len(netparams.train_loader.dataset)
    return train_loss, epoch_acc


def evaluate_model(netparams: NetParams,
                   train_loss,
                   valid_loss,
                   valid_loss_min,
                   valid_loss_array,
                   train_loss_array,
                   epoch,
                   no_improvement):
    print('Evaluation')
    correct_outputs = 0
    netparams.model.eval()
    for batch_i, (data, target) in enumerate(netparams.validation_loader):

        # cuda
        if netparams.train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = netparams.model(data)
        loss = netparams.criterion(output, target)
        valid_loss += loss.item() * data.size(0)

        _, preds = torch.max(output, 1)
        correct_outputs += torch.sum(preds == target.data)

    train_loss = train_loss / len(netparams.train_loader.sampler)
    valid_loss = valid_loss / len(netparams.validation_loader.sampler)
    epoch_acc = float(correct_outputs) / \
        len(netparams.validation_loader.dataset)
    # print training/validation statistics
    print(
        f'Epoch: {epoch}/{netparams.n_epochs} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
    train_loss_array.append(train_loss)
    valid_loss_array.append(valid_loss)
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print(colored(
            f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...', 'green'))
        torch.save(netparams.model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
        no_improvement = 0
    else:
        no_improvement += 1
        print(colored(f'No improvement for {no_improvement} epochs', 'red'))
    return valid_loss_min, no_improvement, epoch_acc


def plot_array(plot_name: str, array: set, best_epoch: int):
    plt.plot(array[:best_epoch])
    plt.ylabel(plot_name)
    plt.show()


def test_model(netparams: NetParams, working_ds: DatasetModel):
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(working_ds.class_num))
    class_total = list(0. for i in range(working_ds.class_num))
    top_k = []

    netparams.model.eval()  # eval mode
    if netparams.train_on_gpu:
        netparams.model.cuda()

    # iterate over test data
    for data, target in netparams.test_loader:

        if len(target.data) < netparams.batch_size:
            break
        # move tensors to GPU if CUDA is available
        if netparams.train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = netparams.model(data)
        # calculate the batch loss
        loss = netparams.criterion(output, target)
        # update  test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not netparams.train_on_gpu else np.squeeze(
            correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(netparams.batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        top_k.append(top_k_1_5(output, target))

    # calculate avg test loss
    test_loss = test_loss / len(netparams.test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    class_accuracy = {}

    for i in range(working_ds.class_num):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                working_ds.classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

            class_accuracy[working_ds.classes[i]
                           ] = class_correct[i] / class_total[i]

        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (working_ds.classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    mean_1, mean_5 = mean_top_k(top_k)
    return 100. * np.sum(class_correct) / np.sum(class_total), mean_1.item(), mean_5.item(), class_accuracy


def plot_test_results(netparams: NetParams, working_ds: DatasetModel):
    # obtain one batch of test images
    dataiter = iter(netparams.test_loader)
    images, labels = dataiter.next()
    images.numpy()

    netparams.model.cpu()
    # get sample outputs
    output = netparams.model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(30, 6))
    for idx in np.arange(netparams.batch_size):
        ax = fig.add_subplot(2, netparams.batch_size / 2,
                             idx + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title("{}\n({})".format(working_ds.classes[preds[idx]], working_ds.classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))


def top_k_1_5(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        # max number labels we will consider in the right choices for out model
        maxk = max(topk)
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        # _, [B, n_classes] -> [B, maxk]
        _, y_pred = output.topk(k=maxk, dim=1)
        # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        y_pred = y_pred.t()

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        # [B] -> [B, 1] -> [maxk, B]
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        # [maxk, B] were for each example we know which topk prediction matched truth
        correct = (y_pred == target_reshaped)
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            # [k, B] -> [kB]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        # list of topk accuracies for entire batch [topk1, topk2, ... etc]
        return list_topk_accs


def mean_top_k(top_k: []) -> tuple:
    sum_1 = 0
    sum_5 = 0

    for i in top_k:
        sum_1 += i[0]
        sum_5 += i[1]

    return sum_1 / len(top_k), sum_5 / len(top_k)


def recall_precision_fmeasure(netparams: NetParams, working_ds: DatasetModel):
    netparams.model.eval()  # eval mode
    if netparams.train_on_gpu:
        netparams.model.cuda()
    i = 0
    f1 = 0.0
    precision = 0.0
    recall = 0.0

    for data, target in netparams.confusion_loader:

        # move tensors to GPU if CUDA is available
        if netparams.train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = netparams.model(data)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        f1 += f1_score(pred.cpu(), target.cpu())
        precision += precision_score(pred.cpu(), target.cpu())
        recall += recall_score(pred.cpu(), target.cpu())
        i+=1

    return f1, precision, recall


def confusion(netparams: NetParams, working_ds: DatasetModel):
    confusion_matrix = torch.zeros(working_ds.class_num, working_ds.class_num)
    with torch.no_grad():
        netparams.model.cpu()
        for i, (inputs, classes) in enumerate(netparams.confusion_loader):
            inputs = inputs.cpu()
            classes = classes.cpu()
            outputs = netparams.model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def loop_inplace_sum(confusions):
    # assumes len(arrlist) > 0
    sum = confusions[0].copy()
    for a in confusions[1:]:
        sum += a
    return sum/len(confusions)


def get_plot_confusion(confusion_array, working_ds: DatasetModel):
    df_cm = pd.DataFrame(confusion_array.numpy(), index=[i for i in working_ds.classes],
                         columns=[i for i in working_ds.classes])
    plt.figure(figsize=(20, 15))
    return sn.heatmap(df_cm, annot=True, cmap='BuPu')
