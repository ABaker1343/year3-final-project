import numpy as np
import GPy
import matplotlib.pyplot as plt

from loader import load_time_series, with_unix, split_dates

dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv")
dataframe = with_unix(dataframe)
dataframe = split_dates(dataframe)
print(dataframe["Day"])
dataframe = dataframe.sample(frac=0.05)

X = dataframe[["Unix", "Close", "Volume"]].to_numpy()
Y = dataframe[["Close"]]

#X = np.random.uniform(-3, 3, (20,1))
#Y = np.sin(X) + np.random.randn(20,1) * 0.05

print(X)
print(dataframe.Unix)

GPy.plotting.change_plotting_library("matplotlib")

# select the kernel for our guassian process
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
kernel = GPy.kern.Cosine(input_dim=1, variance=.1, lengthscale=.1)
kernel = GPy.kern.StdPeriodic(1)
plt.show()

#define the model
model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)
print(model)

fig = model.plot(visible_dims=[0])
#print(fig)
plt.show()
#GPy.plotting.show(fig)

model.optimize(messages=True)
fig = model.plot(visible_dims=[0])
plt.show()
#model.optimize_restarts(num_restarts = 10)
