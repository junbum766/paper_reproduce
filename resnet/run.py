import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
from dataloader import train_data, test_data

model = Resnet18(10).to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

trainloader = train_data()
testloader = test_data()

EPOCHS = 100
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0 and i > 0:
            print(f"Loss [{epoch+1}, {i}](epoch, minibatch): ", running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)

print("Training Done")


correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy on 10,000 test images: ", 100 * (correct / total), "%")