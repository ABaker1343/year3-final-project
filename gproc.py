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
dataframe["Day"] = normalize_dataframe(dataframe["Day"])

# split data so that we can use 20% for predictions

data_length = len(dataframe)
training_percentage = 0.8
training = dataframe.iloc[: int(data_length * training_percentage)]
testing = dataframe.iloc[int(data_length * training_percentage) :]

data_fraction = 0.2
#training = training.sample(frac=data_fraction)
start_index = int(len(training) * (1 - data_fraction))
training = training.iloc[start_index : ]

X = training[["Day"]].to_numpy()
Y = training[["Close"]]

# define the testing data
X_test = testing[["Day"]]
Y_test = testing[["Close"]]

print(X)

# select the kernel for our guassian process
kernels = []
kernels.append(rbf_kernel := GPy.kern.RBF(input_dim=1, variance=1/2, lengthscale=1/12)) # a lower length scale will make it so that values are evaluated as less similar
#kernels.append(linear_kernel := GPy.kern.Linear(input_dim=1, variances=1.0))
kernels.append(periodic_kernel := GPy.kern.StdPeriodic(input_dim=1, variance=1/1, period=1/48))  # frequent period on this kernel to capture the constant up and down
# combine the RBF and linear kernel to try and intergrate a more general linear trend into the RBF kernel
kernel = GPy.kern.Add(kernels)

kernel.plot()

#define the model
model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)

# opimise the model
model.optimize(messages=True)

# train the model multiple times and take the best one
#model.optimize_restarts(num_restarts = 10)
#fig = model.plot(visible_dims=[0])
#plt.show()

num_days_predicted = 30
X_test = X_test.iloc[:num_days_predicted].to_numpy()
Y_test = Y_test.iloc[:num_days_predicted]

# make our predictions
pred_test = model.predict(X_test)

mse = 0
for i in range(num_days_predicted):
    mse += pow(Y_test["Close"].iloc[i] - pred_test[0][i], 2)
    print(Y_test["Close"].iloc[i] , pred_test[0][i])

# plot the model and the predictions

fig = model.plot(visible_dims=[0])
plt.plot(X_test, pred_test[0], "ro")
plt.plot(X_test, Y_test, "go")
plt.show()
#plt.show(filename="gaussian_process_plot.png")

plt.plot(X_test, Y_test["Close"], 'bo')
plt.plot(X_test, pred_test[0], 'ro')
plt.show()

print(f"mse for extrapolated testing data: {mse}")

