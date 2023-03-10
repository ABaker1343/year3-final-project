import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas

import scipy.stats as stats

from loader import *

#dataframe = load_time_series("datasets/tetuan_city_power_consumption.csv").iloc[:200]
num_prev_days = 14
num_future_days = 3

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

training_fraction = 0.8
split_point = int(len(dataframe) * training_fraction)
training = dataframe.iloc[ : split_point]
testing = dataframe.iloc[split_point : split_point + num_future_days]

X = training[params].iloc[num_prev_days:]
Y = training[pred_params].apply(pandas.to_numeric).iloc[num_prev_days:]

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
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/12, in_features=len(params), out_features=len(params) * 5),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=1/12, in_features=len(params) * 5, out_features=num_future_days)
        )

#model = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=len(params), out_features=num_future_days)
#model = nn.Linear(len(params), num_future_days)

# move the data and the model to the device
x = x.to(device)
y = y.to(device)
model = model.to(device)

# select our loss functions and optimiser
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.1
learning_rate = 1.0e-7
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

mse_loss = mse_loss.to(device)
kl_loss = kl_loss.to(device)

# set the number of epochs that we want to use
num_epochs = 50_000
#num_epochs = 10_000

convergance_tolerance = sys.float_info.min #1.0e-3
accuracy_tolerance = sys.float_info.min #1.0e0

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

#def draw_dist(predictions):
    #mu = np.mean(predictions)
    #sigma = np.std(predictions)
    #print(f"mu: {mu} sigma: {sigma}")
    #plt.plot(stats.norm.pdf(mu, sigma))
    #plt.show()

num_tests = 12
for test_num in range(num_tests):

    num_predictions = 10_000
    predictions = []
    prediction_means = []
    prediction_stds = []
    test_data = testing.iloc[test_num]

    x_testing = torch.from_numpy(test_data[params].values).float()
    x_testing = x_testing.to(device)

    y_testing = torch.from_numpy(test_data[pred_params].values).float()

    for i in range(num_predictions):
        predictions.append(model(x_testing).cpu().detach().numpy())

    # find the mean and standard deviation of the predictions
    predictions_trans = np.transpose(predictions)

    for d in range(num_future_days):
        predictions_trans = np.transpose
        prediction_means.append(np.mean(np.transpose(predictions)[d]))
        prediction_stds.append(np.std(np.transpose(predictions)[d]))

    print("prediction means: ", prediction_means)
    print("prediction stds: ", prediction_stds)
    print("real: ", test_data)

#draw_dist([x[0][0] for x in predictions])

#for m in model.modules():
    #if isinstance(m, bnn.BayesLinear):
        #print(f"weight mean: {m.weight_mu}, \nweight sigma: {m.weight_log_sigma}")
        #print(m.extra_repr())

## print predictions out to console
#print("real: ", y_testing, "\n", "predicted: ", predictions)

#for m in model.modules():
    #print (m)

#draw_plot(predictions[0].data)

