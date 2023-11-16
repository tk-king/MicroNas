
def train_pytorch(model, trainloader, optimizer, criterion):
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            time_data, freq_data, labels = data


            optimizer.zero_grad()

            outputs = model(time_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

print('Finished Training')