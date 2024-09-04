import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# 1. load data
## 1.1 use numpy to load the dataset, 
##     split into input (X) and output (y) variables
dataset = np.loadtxt('diabetes.csv', delimiter=',')
X = dataset[:,0:8] # @ shape (768, 8)
y = dataset[:,8] # @ shape (768,)
## 1.2 convert numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32) ## @ shape torch.Size([768, 8])
## - if no reshape, y will be shape torch.Size([768])
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) ## - shape torch.Size([768, 1])

# 2. define model 
## 2.1 with nn.Sequential
model1 = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

## 2.2 with or without nn.ModuleList
# layers = []
layers = nn.ModuleList()
layers.append(nn.Linear(8, 16))
layers.append(nn.ReLU())
layers.append(nn.Linear(16, 32))
layers.append(nn.ReLU())
layers.append(nn.Linear(32, 1))
layers.append(nn.Sigmoid())
model2 = nn.Sequential(*layers)

## 2.3 with nn.Module
class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x
    
model3 = Model(8)


# 2.4 maybe add the manual seed for randomization
# torch.manual.seed(1234)
# 3. define loss function and optimizer
loss_function = nn.BCELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# 4. train the model
num_epochs = 100
batch_size = 16

for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        # * forward pass
        y_pred = model1(X[i:i+batch_size])
        loss = loss_function(y_pred, y[i:i+batch_size])
        
        # * backward pass
        ## - 1. zero the gradients
        ## - 2. perform a backward pass (backpropagation)
        ## - 3. update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# 5. evaluate the model
##   compute accuracy (no_grad is optional but saves memory and suggested)
## -  torch.no_grad() 是一个上下文管理器，用于在PyTorch中禁用梯度计算的上下文中。
## -  它可以视为一个资源，它临时地关闭了梯度计算功能。  
with torch.no_grad():
    y_pred = model1(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# 6. Make predictions
# make a prediction
## - input_data must be a tensor
## - from list to tensor
input_data = torch.tensor([6,148,72,35,0,33.6,0.627,50])
y_pred = model1(input_data)
print(f"Prediction: {y_pred.item():.4f}")

# 7. save and load model
## 7.1 save model
torch.save(model1.state_dict(), 'model.pt')

## 7.2 load model
model = model1
model.load_state_dict(torch.load('model.pt'))
model.eval()

##  7.3 make a prediction
input_data = torch.tensor([6,148,72,35,0,33.6,0.627,50])
y_pred = model(input_data)
print(f"Prediction: {y_pred.item():.4f}")

# 8. Tensorboard
writer = SummaryWriter("./torchlogs/")
model = Model(8)
writer.add_graph(model, X)
writer.close()