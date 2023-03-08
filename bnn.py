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
dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv", fields + pred_fields,
                             num_prev_days=num_prev_days, prev_fields=["Close"],
                             num_future_days=num_future_days - 1, future_fields=["Close"])

dataframe["Day"] = normalize_dataframe(dataframe["Day"])

# create the x array based on the number of previous days
params = ["Day"]
for i in range(1, num_prev_days):
    params.append("Close" + str(i) + "Day")

pred_params = ["Close"]
for i in range(1, num_future_days):
    pred_params.append("Close" + "+" + str(i) + "Day")

print(dataframe)

training_fraction = 0.8
split_point = int(len(dataframe) * training_fraction)
training = dataframe.iloc[ : split_point]
testing = dataframe.iloc[split_point : split_point + num_future_days]

X = training[params].iloc[num_prev_days:]
Y = training[pred_params].apply(pandas.to_numeric).iloc[num_prev_days:]

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
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=len(params), out_features=len(params) * 10),
        #nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/3, in_features=len(params) * 10, out_features=num_future_days)
        )

#model = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=1)
#model = nn.Linear(len(params), num_future_days)

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
num_epochs = 30_000
#num_epochs = 1_000

convergance_tolerance = 1.0e-10
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

print(f"final loss -- {cost}")

# plot the results
def draw_plot(predicted) :
    days = testing['Day']
    predicted = predicted.cpu()
    plt.plot(days, testing[pred_params], 'bo')
    plt.plot(days, predicted, 'ro')
    plt.xlabel("time")
    plt.ylabel("closing price")

    plt.show()

x_testing = torch.from_numpy(testing[params].values).float()
x_testing = x_testing.to(device)
y_testing = torch.from_numpy(testing[pred_params].values).float()

for i in range(10):
    predictions = model(x_testing)
    prediction = predictions

## print predictions out to console
    print(model.weight_mu)
    print("real: ", y_testing, "\n", "predicted: ", predictions)

draw_plot(predictions.data)

