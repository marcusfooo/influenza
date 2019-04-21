import torch
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt

def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc):
    plt.style.use('ggplot')
    
    plt.figure()
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.figure()
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.show()


def get_predictions(scores):
    prob = F.softmax(scores, dim=1)
    _, predictions = torch.topk(prob, 1)
    return predictions


def confusion_matrix(y_true, y_pred):
    y_pred = y_pred.view_as(y_true)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    return TP, FP, TN, FN
            

def train_rnn(model, epochs, learning_rate, batch_size, X, Y, X_test, Y_test):
    print_interval = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples/batch_size)

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_val_accs = []

    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0
        running_acc = 0

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size, batch_size):
            optimizer.zero_grad()
            repackage_hidden(hidden)

            X_batch = X[:, count:count+batch_size, :]
            Y_batch = Y[count:count+batch_size]

            scores = model(X_batch, hidden)
            loss = criterion(scores, Y_batch)
            loss.backward()
            optimizer.step()

            predictions = get_predictions(scores)
            TP, _, TN, _ = confusion_matrix(Y_batch, predictions)
            running_acc += TP + TN

            running_loss += loss.item()

        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            scores = model(X_test, model.init_hidden(Y_test.shape[0]))
            predictions = get_predictions(scores)
            
            TP, FP, TN, FN = confusion_matrix(Y_test, predictions)
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            
            val_acc = (TP + TN) / (TP + FP + TN + FN)
            all_val_accs.append(val_acc)

            val_loss = criterion(scores, Y_test).item()
            all_val_losses.append(val_loss)

        if epoch % print_interval == 0:
            print(' Epoch %d\tTime %.0f s\tLoss %.3f\tAcc  %.3f\tV loss %.3f\tV acc  %.3f\tPrecis %.3f\tRecall  %.3f'
                % (epoch, elapsed_time, epoch_loss, epoch_acc, val_loss, val_acc, precision, recall))

    plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs)