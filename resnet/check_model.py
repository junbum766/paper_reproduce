import torchsummary
from resnet import Resnet34, Resnet50

model1 = Resnet34(1000).to('cuda')
model2 = Resnet50(1000).to('cuda')

print("Resnet34 architecture: ")
torchsummary.summary(model1, (3, 224, 224))
print()

print("Resnet50 architecture: ")
torchsummary.summary(model2, (3, 224, 224))