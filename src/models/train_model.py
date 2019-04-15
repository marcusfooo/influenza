import torch
import math
import time

def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

def train_rnn(model,epochs, learning_rate, batch_size, X, Y):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_function = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples/batch_size)

    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size, batch_size):
            optimizer.zero_grad()
            repackage_hidden(hidden)

            X_batch = X[:, count:count+batch_size, :]
            Y_batch = Y[count:count+batch_size]

            scores = model(X_batch, hidden)
            loss = loss_function(scores, Y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss/num_of_batches
        elapsed_time = time.time()-start_time


        print('epoch=', epoch, '\t time=', elapsed_time, '\t loss',  epoch_loss)