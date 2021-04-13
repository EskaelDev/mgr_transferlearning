import matplotlib.pyplot as plt
from netparams import NetParams
import torch
from datasetmodel import DatasetModel
import numpy as np
import time
from termcolor import colored
from trainstats import TrainStats
import torch.nn.functional as F


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

    end_time = time.time()
    print(
        f"🎓Total learning took {((end_time - start_time)/60):.0f}m {((end_time - start_time)%60):.0f}s")
    print(
        f"🏋️‍♂️Training took {(train_time_sum/60):02.0f}:{(train_time_sum%60):02.3f}")
    print(
        f"📑Evaluation took {(eval_time_sum/60):02.0f}:{(eval_time_sum%60):02.3f}")
    reults = TrainStats(netparams.model.name, train_loss_array, valid_loss_array, train_accuracy_array,
                        valid_accuracy_array, best_epoch, end_time - start_time, train_time_sum, eval_time_sum)
    return reults


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
        torch.save(netparams.model.state_dict(), f'{epoch:03d}model_cifar.pt')
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

    # calculate avg test loss
    test_loss = test_loss / len(netparams.test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(working_ds.class_num):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                working_ds.classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (working_ds.classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    return 100. * np.sum(class_correct) / np.sum(class_total)


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
