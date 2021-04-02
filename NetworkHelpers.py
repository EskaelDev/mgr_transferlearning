def train_loop(criterion,
               optimizer,
               n_epochs,
               no_improvement,
               max_no_improve_epochs):

    valid_loss_min = np.Inf
    train_loss_array = []
    valid_loss_array = []

    for epoch in range(1, n_epochs+1):

        # early stopping
    if no_improvement >= max_no_improve_epochs:
        break

    # keep track of training and validation loss
    train_loss = 0.0
    train_loss_tmp = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    ######################
    # evaluate the model #
    ######################


def train_model(model,
                train_loader,
                optimizer,
                criterion,
                train_loss_tmp,
                train_loss,
                epoch,
                n_epochs,
                train_on_gpu):
    model.train()
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss_tmp += loss.item()
        train_loss += loss.item() * data.size(0)

        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print(
                f'Epoch {epoch}, Batch {batch_i + 1}/{batch_size} loss: {(train_loss_tmp / 20):.16f}')
            train_loss_tmp = 0.0


def evaluate_model(model,
                   validation_loader,
                   train_loss,
                   valid_loss
                   criterion,
                   valid_loss_min,
                   valid_loss_array,
                   train_loss_array,
                   epoch,
                   n_epochs,
                   train_on_gpu):
    model.eval()
    for batch_i, (data, target) in enumerate(validation_loader):

        # cuda
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        y = model(data)
        loss = criterion(y, target)
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(validation_loader.sampler)

    # print training/validation statistics
    print(
        f'Evaluation\nEpoch: {epoch + 1}/{n_epochs+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
    train_loss_array.append(train_loss)
    valid_loss_array.append(valid_loss)
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print(
            f'⌛ Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'{epoch:03d}model_cifar.pt')
        valid_loss_min = valid_loss
        no_improvement = 0
    else:
        no_improvement += 1


def plot_loss(loss_name: str, loss_array: set):
    min_idx = loss_array.index(min(loss_array))
    plt.plot(loss_array[:min_idx])
    plt.ylabel(loss_name)
    plt.show()


def test_model(classes,
               test_loader,
               train_on_gpu,
               model,
               criterion):
    # track test loss
    # over 5 flower classes
    test_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    model.eval()  # eval mode
    if train_on_gpu:
        model.cuda()

    # iterate over test data
    for data, target in test_loader:

        if len(target.data) < batch_size:
            break
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update  test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
            correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(len(class_total)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
