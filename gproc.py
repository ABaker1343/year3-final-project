import numpy as np
import GPy
import matplotlib.pyplot as plt

from loader import load_time_series, normalize_dataframe, with_unix, split_dates, with_days

GPy.plotting.change_plotting_library("matplotlib")

dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv")
dataframe = with_unix(dataframe)
dataframe = split_dates(dataframe)
dataframe = with_days(dataframe)

# strip out the fields that we dont want
dataframe = dataframe[["Day", "Close"]]

# normalize data in the dataframe
dataframe = normalize_dataframe(dataframe)

data_length = len(dataframe)
# split data so that we can use 20% for predictions
training_percentage = 0.8
training = dataframe.iloc[: int(data_length * training_percentage)]
testing = dataframe.iloc[int(data_length * training_percentage) :]

training = training.sample(frac=0.2)

X = training[["Day"]].to_numpy()
Y = training[["Close"]]

print(X)

# select the kernel for our guassian process
kernels = []
kernels.append(rbf_kernel := GPy.kern.RBF(input_dim=1, variance=1/2, lengthscale=1/12)) # a lower length scale will make it so that values are evaluated as less similar
#kernels.append(linear_kernel := GPy.kern.Linear(input_dim=1, variances=1.0))
kernels.append(periodic_kernel := GPy.kern.StdPeriodic(input_dim=1, variance=1/1, period=1/12))
# combine the RBF and linear kernel to try and intergrate a more general linear trend into the RBF kernel
kernel = GPy.kern.Add(kernels)

kernel.plot()

#define the model
model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)

#fig = model.plot(visible_dims=[0])
#print(fig)
#GPy.plotting.show(fig)

model.optimize(messages=True)
fig = model.plot(visible_dims=[0])
plt.show()
# train the model multiple times and take the best one
#model.optimize_restarts(num_restarts = 10)
#fig = model.plot(visible_dims=[0])
#plt.show()

# test the model with data that we have remaining
# and some new data that we have to extrapolate
X_test = testing[["Day"]]
Y_test = testing[["Close"]]

num_days_predicted = 14
X_test = X_test.iloc[:num_days_predicted].to_numpy()
Y_test = Y_test.iloc[:num_days_predicted]

pred_test = model.predict(X_test)

mse = 0
for i in range(num_days_predicted):
    mse += pow(Y_test["Close"].iloc[i] - pred_test[0][i], 2)
    print(Y_test["Close"].iloc[i] , pred_test[0][i])

plt.plot(X_test, Y_test["Close"], 'bo')
plt.plot(X_test, pred_test[0], 'ro')
plt.show()

print(f"mse for extrapolated testing data: {mse}")
