import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas

from loader import *

#dataframe = load_time_series("datasets/tetuan_city_power_consumption.csv").iloc[:200]
num_prev_days = 14
num_future_days = 1

fields=["Day"]
pred_fields = ["Close"]
dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv", fields + pred_fields, num_prev_days=num_prev_days, prev_fields=["Close"])

dataframe["Day"] = normalize_dataframe(dataframe["Day"])

# create the x array based on the number of previous days
params = ["Day"]
print(dataframe)
for i in range(1, num_prev_days):
    params.append("Close" + str(i) + "Day")

X = dataframe[params].iloc[num_prev_days:]
Y = dataframe[pred_fields].apply(pandas.to_numeric).iloc[num_prev_days:]

print(X)
print(Y)

# get the gpu to send the model to
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

device = torch.device(device_type)
print(f"device type set to {device_type}")

# create the tensors
x, y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()

# create the model for our bayesian neural network

model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=len(params), out_features=len(params) * 5),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=len(params) * 5, out_features=1)
        )

#model = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=1)
#model = nn.Linear(len(params),1)

# move the data and the model to the device
x = x.to(device)
y = y.to(device)
model = model.to(device)

# select our loss functions and optimiser
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0 # 0.1
learning_rate = 1.0e-5
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

mse_loss = mse_loss.to(device)
kl_loss = kl_loss.to(device)

# set the number of epochs that we want to use
num_epochs = 20_000
#num_epochs = 1_000

convergance_tolerance = 1.0e-9
accuracy_tolerance = sys.float_info.min # 1.0e-6

# train our model
prev_loss = sys.float_info.max

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

    if abs(prev_loss - cost) < convergance_tolerance or cost < accuracy_tolerance:
        break

    if device_type == "cpu":
        print(f"epoch {i} -- loss {cost}")
    elif i % 5000 == 0:
        print(f"epoch {i} -- loss {cost}")

    prev_loss = cost

print("final loss -- {cost}")

# plot the results
def draw_plot(predicted) :
    days = dataframe['Day'].iloc[num_prev_days:]
    predicted = predicted.cpu()
    plt.plot(days, Y, 'bo')
    plt.plot(days, predicted, 'ro')
    plt.xlabel("time")
    plt.ylabel("closing price")

    plt.show()

predictions = model(x)
prediction = predictions
draw_plot(predictions.data)

