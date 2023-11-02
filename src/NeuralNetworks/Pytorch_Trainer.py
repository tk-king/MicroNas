
import torch
from tqdm import tqdm

def trainModel(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs=10):
    last_loss = 0.


    for epoch_index in range(epochs):
        train_accu = []
        train_losses = []
        correct=0
        total=0
        running_loss = 0.

        for i, (input_time, input_freq, target) in enumerate(tqdm(train_dataloader)):
            # Every data instance is an input + label pair

            # input_time = torch.unsqueeze(input_time, dim=1)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_time.float())

            # Compute the loss and its gradients
            loss = loss_fn(outputs, target)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            # if i % 100 == 99:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(train_dataloader) + i + 1
            #     print('Loss/train', last_loss, tb_x)
            #     running_loss = 0.

            total += target.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()

        train_loss=running_loss/len(train_dataloader)
        accu=100.*correct/total
        train_accu.append(accu)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Train_Accuracy: %.3f'%(train_loss,accu))

        # Test the model after each epoch with the test_data_loader
        test_accu = []
        correct=0
        total=0
        for i, (input_time, input_freq, target) in enumerate(tqdm(test_dataloader)):
            input_time = torch.unsqueeze(input_time, dim=1)
            total += target.size(0)
            outputs = model(input_time.float())
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            accu=100.*correct/total
            test_accu.append(accu)
        print('Test_Accuracy: %.3f'%(accu))