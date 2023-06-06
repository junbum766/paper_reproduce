import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
from dataloader import train_data, test_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=int, default=18, choices=[18, 34, 50, 101, 152],help="select a model among Resnet series (18, 34, 50, 101, 152)")
parser.add_argument('--epoch', type=int, default=100)
args = parser.parse_args()

print('you choose the Resnet{} model!'.format(args.model))

if args.model == 18:
    model = Resnet18(10).cuda()
elif args.model == 34:
    model = Resnet34(10).cuda()
elif args.model == 50:
    model = Resnet50(10).cuda()
elif args.model == 101:
    model = Resnet101(10).cuda()
elif args.model == 152:
    model = Resnet152(10).cuda()
else:
    print('error: please check a model argument.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

trainloader = train_data()
testloader = test_data()

EPOCHS = args.epoch
for epoch in range(EPOCHS):
    print('epoch', epoch+1, ': ')
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('outputs shape: ', outputs.shape)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        # print('loss: ', loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print('running_loss: ', running_loss)
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
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy on 10,000 test images: ", 100 * (correct / total), "%")