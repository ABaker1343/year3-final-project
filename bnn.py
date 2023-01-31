import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
import pandas
import time

iris = datasets.load_iris()
X = iris.data
Y = iris.target

dataframe = pandas.read_csv("~/datasets/stock_market_data/sp500/csv/AMD.csv")

unix_times = [(pandas.to_datetime(row['Date']) - pandas.Timestamp("1970-01-01")) // pandas.Timedelta('1s') for _, row in dataframe.iterrows()]
    
dataframe['UNIX'] = unix_times

X = dataframe[['UNIX', 'Open']].apply(pandas.to_numeric)
Y = dataframe['Close'].apply(pandas.to_numeric)

print(dataframe)

#x, y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
x, y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float()
print(x.shape, y.shape)

# create the model for our bayesian neural network

model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=8),
        #nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=8, out_features=1)
        )

model = nn.Linear(2,1)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimiser = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 3000

for i in range(num_epochs):
    predictions = model(x)
    mse = mse_loss(predictions, y)
    kl = kl_loss(model)
    cost = mse + kl_weight * kl

    optimiser.zero_grad()
    # back propigate the loss
    cost.backward()
    
    optimiser.step()

    print(f"epoch {i} -- loss {cost}")

_, predicted = torch.max(predictions.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
#print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))


# plot the model

def draw_plot(predicted) :
    plt.plot(X['UNIX'], Y, 'bo')
    plt.plot(X['UNIX'], predicted, 'ro')

    plt.show()

predictions = model(x)
print(predictions.data)
draw_plot(predictions.data)

