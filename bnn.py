import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas

from loader import load_time_series, with_unix, split_dates

dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv")

# change the dates into unix timestamps as that is a numeric type
dataframe = with_unix(dataframe)
dataframe = split_dates(dataframe)
# select the data that will be used to train our model
#dataframe = dataframe.sample(frac=0.4)

X = dataframe[['High', 'Volume', "Close"]].apply(pandas.to_numeric)
Y = dataframe['Close'].apply(pandas.to_numeric)

# get the gpu to send the model to
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

device = torch.device(device_type)
print(f"device type set to {device_type}")

# create the tensors
print(X.values)
x, y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
print(x.shape, y.shape)

# create the model for our bayesian neural network

model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=3, out_features=300),
        #nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=300, out_features=1)
        )

model = nn.Linear(3,1)

# move the data and the model to the device
x = x.to(device)
y = y.to(device)
model = model.to(device)

# select our loss functions and optimiser
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimiser = optim.Adam(model.parameters(), lr=0.01)

mse_loss = mse_loss.to(device)
kl_loss = kl_loss.to(device)

# set the number of epochs that we want to use
num_epochs = 10_000
#num_epochs = 1_000

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

# plot the results
def draw_plot(predicted) :
    predicted = predicted.cpu()
    plt.plot(dataframe['Unix'], Y, 'bo')
    plt.plot(dataframe['Unix'], predicted, 'ro')
    print(predicted)
    plt.xlabel("time")
    plt.ylabel("closing price")

    plt.show()

predictions = model(x)
prediction = predictions
draw_plot(predictions.data)

