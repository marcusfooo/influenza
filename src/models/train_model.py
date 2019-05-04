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


def plot_training_history(loss, val_loss, acc, val_acc, mini_batch_scores, mini_batch_labels):
    plt.style.use('ggplot')
    
    # Plot losses
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1,3,2)
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()

    # Plot prediction dynamics of test mini batch
    plt.subplot(1,3,3)
    for i in range(len(mini_batch_labels)):
        if mini_batch_labels[i]:
            score_sequence = [x[i][1] for x in mini_batch_scores]
            plt.plot(score_sequence, 'b', label='Pos')
        else:
            score_sequence = [x[i][0] for x in mini_batch_scores]
            plt.plot(score_sequence, 'r', label='Neg')
    
    plt.title('Logits')
    plt.legend(labels=['Pos', 'Neg'])
    plt.show()


def predictions_from_output(scores):
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def get_confusion_matrix(y_true, y_pred):
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

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


def verify_model(model, X, Y, criterion):
    X.requires_grad_()
    criterion = torch.nn.CrossEntropyLoss()
    scores = model(X, model.init_hidden(Y.shape[0]))
    print('Loss @ init: %.3f' % criterion(scores, Y).item())

    non_zero_idx = 1
    perfect_scores = [[0, 0] for y in Y]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i, y in enumerate(Y)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    criterion = torch.nn.MSELoss()
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
                if sum(X.grad[i, j] != zero_tensor):
                    assert(j == non_zero_idx)
                else:
                    assert(j != non_zero_idx)

    print('Backpropagated dependencies OK')
    X.detach()


def train_rnn(model, epochs, learning_rate, batch_size, X, Y, X_test, Y_test):
    print_interval = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples/batch_size)

    verify_model(model, X, Y, criterion)

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_val_accs = []

    X_test_mini_batch = X_test[:, 184:192, :]
    Y_test_mini_batch = Y_test[184:192]
    mini_batch_scores = []

    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0
        running_acc = 0

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            optimizer.zero_grad()
            repackage_hidden(hidden)

            X_batch = X[:, count:count+batch_size, :]
            Y_batch = Y[count:count+batch_size]

            scores = model(X_batch, hidden)
            loss = criterion(scores, Y_batch)
            loss.backward()
            optimizer.step()

            predictions = predictions_from_output(scores)
            conf_matrix = get_confusion_matrix(Y_batch, predictions)
            TP, TN = conf_matrix[0][0], conf_matrix[1][1]
            running_acc += TP + TN

            running_loss += loss.item()

        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            scores = model(X_test, model.init_hidden(Y_test.shape[0]))
            predictions = predictions_from_output(scores)
            
            conf_matrix = get_confusion_matrix(Y_test, predictions)
            TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            
            val_acc = (TP + TN) / (TP + FP + TN + FN)
            all_val_accs.append(val_acc)

            val_loss = criterion(scores, Y_test).item()
            all_val_losses.append(val_loss)

            mini_batch_scores.append(model(X_test_mini_batch, model.init_hidden(Y_test_mini_batch.shape[0])))


        if epoch % print_interval == 0:
            print(' Epoch %d\tTime %.0f s\tLoss %.3f\tAcc  %.3f\tV loss %.3f\tV acc  %.3f\tPrecis %.3f\tRecall  %.3f'
                % (epoch, elapsed_time, epoch_loss, epoch_acc, val_loss, val_acc, precision, recall))

    plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, mini_batch_scores, Y_test_mini_batch)