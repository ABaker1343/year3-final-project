import numpy as np
import GPy
import matplotlib.pyplot as plt

from loader import load_time_series, with_unix, split_dates

dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv")
dataframe = with_unix(dataframe)
dataframe = split_dates(dataframe)
print(dataframe["Day"])
dataframe = dataframe.sample(frac=0.05)

X = dataframe[["Unix", "Day", "Month", "Year"]].to_numpy()
Y = dataframe[["Close"]]

print(X)

# select the kernel for our guassian process
kernel = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)

#define the model
model = GPy.models.GPRegression(X, Y, kernel)
print(model)

GPy.plotting.change_plotting_library("matplotlib")
fig = model.plot(visible_dims=[0])
#print(fig)
plt.show()
#GPy.plotting.show(fig)

model.optimize(messages=True)
fig = model.plot(visible_dims=[0])
plt.show()
#model.optimize_restarts(num_restarts = 10)
