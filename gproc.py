import numpy as np
import GPy
import matplotlib.pyplot as plt

from loader import *

GPy.plotting.change_plotting_library("matplotlib")

############################## DATAFRAME SETUP #####################################

dataset = "AMD stock"
dataset= "tetuan power"

fields = ["Day"]

if dataset == "AMD stock":
    pred_fields = ["Close"]
    dataframe = load_time_series("datasets/stock_market_data/sp500/csv/AMD.csv", fields + pred_fields)
else:
    pred_fields = ["Zone 1 Power Consumption"]
    dataframe = load_time_series("datasets/tetuan_city_power_consumption.csv", fields + pred_fields, date_field="DateTime")

# splot our data into training and testing
data_length = len(dataframe)
training_percentage = 0.8
split_point = int(data_length * training_percentage)

training_amount = 14
testing_amount = 2

# normalize the data and split into training and testing
dataframe = dataframe.iloc[split_point - training_amount : split_point + testing_amount]
dataframe = normalize_dataframe(dataframe)

training = dataframe.iloc[ : training_amount]
testing = dataframe.iloc[training_amount : ]
print(testing)

X = training[fields].to_numpy()
Y = training[pred_fields]

# define the testing data
X_test = testing[fields].to_numpy()
Y_test = testing[pred_fields]

print(X)

################################### DEFINE THE MODEL ###############################################

# select the kernel for our guassian process
kernels = []

##################### kernels for stock predictions ###################################
if dataset == "AMD stock":
    kernels.append(rbf_kernel := GPy.kern.RBF(input_dim=len(fields), variance=1, lengthscale=1/2)) 
    # a lower length scale will make it so that values are evaluated as less similar

    #kernels.append(linear_kernel := GPy.kern.Linear(input_dim=len(fields), variances=1.0))

    kernels.append(periodic_kernel := GPy.kern.StdPeriodic(input_dim=len(fields), variance=1/12, period=1/1, lengthscale=1/4))
    # frequent period on this kernel to capture the constant up and down

    # combine the RBF and linear kernel to try and intergrate a more general linear trend into the RBF kernel

##################### kernels for power consumption predictions #######################
else:
    kernels.append(rbf_kernel := GPy.kern.RBF(input_dim=len(fields), variance=1.2, lengthscale=1/10)) 
    #kernels.append(linear_kernel := GPy.kern.Linear(input_dim=len(fields), variances=1.0))
    #kernels.append(periodic_kernel := GPy.kern.StdPeriodic(input_dim=len(fields), variance=1/12, period=1, lengthscale=1/4))
    # combine the RBF and linear kernel to try and intergrate a more general linear trend into the RBF kernel

kernel = GPy.kern.Add(kernels)
kernel.plot(visible_dims=[0])

#define the model
model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)
model.plot(visible_dims=[0])
plt.show()

# opimise the model
model.optimize(messages=True)

# train the model multiple times and take the best one
#model.optimize_restarts(num_restarts = 10)
#fig = model.plot(visible_dims=[0])
#plt.show()

####################################### PLOT THE RESULTS #############################################

# make our predictions
pred_test = model.predict(X_test)

mse = 0
for i in range(len(pred_test)):
    mse += pow(Y_test.iloc[i] - pred_test[0][i], 2)
    print(Y_test.iloc[i] , pred_test[0][i])

# plot the model and the predictions

fig = model.plot(visible_dims=[0])
plt.plot(X_test, pred_test[0], "ro")
plt.plot(X_test, Y_test, "go")
plt.show()
#plt.show(filename="gaussian_process_plot.png")

#plt.plot(X_test, Y_test["Close"], 'bo')
#plt.plot(X_test, pred_test[0], 'ro')
#plt.show()

print(f"mse for extrapolated testing data: {mse}")

