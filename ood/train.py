import numpy as np


def train(train_dataloader, test_dataloader, model, optimizer, scheduler, criterion, metric, train_epoch, metric_freq, device):
    model.train()

    for epoch in range(train_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += loss.item()
            if i % metric_freq == 0:
                accuracy_train = metric(model, train_dataloader, device)
                accuracy_test = metric(model, test_dataloader, device)
                train_loss = str(np.round(running_loss/(i+1), 5)).ljust(7, '0')
                train_accuracy = str(np.round(accuracy_train, 5)).ljust(7, '0')
                test_accuracy = str(np.round(accuracy_test, 5)).ljust(7, '0')

                print (f"epoch: {epoch}, train loss: {train_loss},  train_accuracy: {train_accuracy}, test_accuracy {test_accuracy}")
        scheduler.step()
        print(f'current LR: {scheduler.get_lr()[0]}')