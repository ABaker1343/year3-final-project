import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas
import time

dataframe = pandas.read_csv("~/datasets/stock_market_data/sp500/csv/AMD.csv")

# change the dates into unix timestamps as that is a numeric type
unix_times = [(pandas.to_datetime(row['Date']) - pandas.Timestamp("1970-01-01")) // pandas.Timedelta('1s') for _, row in dataframe.iterrows()]

# select the data that will be used to train our model
dataframe['UNIX'] = unix_times

X = dataframe[['UNIX', 'Open']].apply(pandas.to_numeric)
Y = dataframe['Close'].apply(pandas.to_numeric)

# get the gpu to send the model to
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

print(f"device type set to {device_type}")

device = torch.device(device_type)

# create the tensors
x, y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
print(x.shape, y.shape)

# create the model for our bayesian neural network

model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=2, out_features=8),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=1)
        )

#model = nn.Linear(2,1)

# move the data and the model to the device
x.to(device)
y.to(device)
model.to(device)

# select our loss functions and optimiser
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimiser = optim.Adam(model.parameters(), lr=0.01)

# set the number of epochs that we want to use
num_epochs = 3000

# train our model
for i in range(num_epochs):
    # calculate the loss for the current epoch
    predictions = model(x)
    mse = mse_loss(predictions, y)
    kl = kl_loss(model)
    cost = mse + kl_weight * kl

    # back propagate our results
    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    print(f"epoch {i} -- loss {cost}")

# calculate the accuracy of our model
_, predicted = torch.max(predictions.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
#print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))


# plot the results
def draw_plot(predicted) :
    plt.plot(X['UNIX'], Y, 'bo')
    plt.plot(X['UNIX'], predicted, 'ro')

    plt.show()

predictions = model(x)
print(predictions.data)
draw_plot(predictions.data)

