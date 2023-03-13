#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
import cv2
import numpy as np
from utils import dataset_loader


data = DatasetFolder("./samples", loader=dataset_loader, extensions="avi")

# video = data[0][0]

# print(type(video))
# for i in range(video.shape[0]):
#     cv2.imshow('Frame',video[i,:,:,:])

#     # Press Q on keyboard to  exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

#X = np.stack([x[0] for x in data], axis=0)
#Y = np.stack([x[1] for x in data], axis=0)

print(f'data shape: {data[0][0].shape}')

train_loader = DataLoader(dataset=data,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data

# Dummy Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Run your training process
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using {device} device")

# # Hyper-parameters 
# num_epochs = 5
# batch_size = 4
# learning_rate = 0.001


# # Define model
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # -> n, 3, 32, 32
#         x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
#         x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
#         x = x.view(-1, 16 * 5 * 5)            # -> n, 400
#         x = F.relu(self.fc1(x))               # -> n, 120
#         x = F.relu(self.fc2(x))               # -> n, 84
#         x = self.fc3(x)                       # -> n, 10
#         return x

# model = ConvNet().to(device)
# print(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# n_total_steps = len(data)
# for epoch in range(num_epochs):
#     for images, labels in data:
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         # input_layer: 3 input channels, 6 output channels, 5 kernel size
#         images = images#.to(device)
#         labels = labels#.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 2000 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
