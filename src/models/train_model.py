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

def train_rnn(model, epochs, learning_rate, batch_size, X, Y, X_test, Y_test):
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

            prob = F.softmax(scores, dim=1)
            _, prediction = torch.topk(prob, 1)
            prediction = prediction.view_as(Y_batch)
            running_acc += int((Y_batch.eq(prediction)).sum())

            running_loss += loss.item()

        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            scores = model(X_test, model.init_hidden(Y_test.shape[0]))
            prob = F.softmax(scores, dim=1)
            _, prediction = torch.topk(prob, 1)
            prediction = prediction.view_as(Y_test)

            correct = (Y_test.eq(prediction)).sum()
            val_acc = int(correct)/Y_test.shape[0]
            all_val_accs.append(val_acc)

            val_loss = criterion(scores, Y_test).item()
            all_val_losses.append(val_loss)

        print('Epoch: %d\tTime: %.1f s\tLoss: %.4f\tAcc: %.4f\tVal loss: %.4f\tVal acc: %.4f'
            % (epoch, elapsed_time, epoch_loss, epoch_acc, val_loss, val_acc))
        
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(all_losses, 'b', label='Training')
    plt.plot(all_val_losses, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.figure()
    plt.plot(all_accs, 'b', label='Training')
    plt.plot(all_val_accs, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.show()