import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas

from loader import *

#dataframe = load_time_series("datasets/tetuan_city_power_consumption.csv").iloc[:200]
dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv")
num_prev_days = 3

# change the dates into unix timestamps as that is a numeric type
#dataframe = with_unix(dataframe, "DateTime") # power consumption data
dataframe = with_days(dataframe, "Date")
dataframe = add_prev_days(dataframe, "Day", num_prev_days)

# strip out only what we are using for training
dataframe = dataframe[["High", "Volume", "Day", "Day1Day", "Day2Day", "Open", "Close"]]
dataframe = normalize_dataframe(dataframe)

#X = dataframe[["Humidity", "Temperature", "general diffuse flows"]].apply(pandas.to_numeric).iloc[num_prev_days:]
#Y = dataframe['Zone 2  Power Consumption'].apply(pandas.to_numeric).iloc[num_prev_days:]

X = dataframe[["Day", "Day1Day", "Day2Day"]].apply(pandas.to_numeric).iloc[num_prev_days:]
Y = dataframe['Close'].apply(pandas.to_numeric).iloc[num_prev_days:]
#X = dataframe[["Open", "Volume", "High"]].apply(pandas.to_numeric).iloc[num_prev_days:]

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
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=3, out_features=80),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=80, out_features=1)
        )

#model = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=1)
#model = nn.Linear(3,1)

# move the data and the model to the device
x = x.to(device)
y = y.to(device)
model = model.to(device)

# select our loss functions and optimiser
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimiser = optim.Adam(model.parameters(), lr=1.0e-5)

mse_loss = mse_loss.to(device)
kl_loss = kl_loss.to(device)

# set the number of epochs that we want to use
num_epochs = 25_000
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

    if i % 5000 == 0:
        print(f"epoch {i} -- loss {cost}")

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

