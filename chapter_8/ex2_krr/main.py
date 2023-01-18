import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
import joblib

krr = KernelRidge(kernel='rbf')
krr_params = {"alpha": np.arange(0, 0.02, 0.005),
              "gamma": np.arange(3, 4, 0.1)}

# Read text input
file_name = "wave.csv"
db = pd.read_csv(file_name, delimiter=',', header=0)

print(db.head(5))

# Split data into train/test
x_train, x_test, y_train, y_test = train_test_split(db.x, db.y, test_size=0.2)

# save splitted dataframe
db_train = pd.DataFrame((x_train, y_train)).T
db_test = pd.DataFrame((x_test, y_test)).T
db_train.to_csv("train.csv", index=False)
db_test.to_csv("test.csv", index=False)

# run model optimizer
hypermodel = GridSearchCV(krr, krr_params, cv=5, scoring="neg_mean_squared_error")
x_train = np.asarray(x_train).reshape(-1, 1)
hypermodel.fit(x_train, y_train)

print(hypermodel.best_params_)

# validate results
x_test = np.asarray(x_test).reshape((-1, 1))
y_pred = hypermodel.predict(x_test)
y_tr_pred = hypermodel.predict(x_train)
r2 = metrics.r2_score(y_pred, y_test)
mse = metrics.mean_squared_error(y_pred, y_test)
mae = metrics.mean_absolute_error(y_pred, y_test)
r2_tr = metrics.r2_score(y_tr_pred, y_train)
mse_tr = metrics.mean_squared_error(y_tr_pred, y_train)
mae_tr = metrics.mean_absolute_error(y_tr_pred, y_train)

# print(r2)
# print(mse)
# print(mae)
# save scores
with open('scores.json', 'w') as f:
    f.write('{\n')
    f.write(f'\t"test_mae" : {mae},\n')
    f.write(f'\t"test_mse" : {mse},\n')
    f.write(f'\t"test_r2" : {r2},\n')
    f.write(f'\t"train_mae" : {mae_tr},\n')
    f.write(f'\t"train_mse" : {mse_tr},\n')
    f.write(f'\t"train_r2" : {r2_tr}\n')
    f.write('}')

# Plot results
x_space = np.arange(-10, 10, 0.01)
y_f = np.exp(-1*np.square(x_space/4))*np.cos(4*x_space)
y_f_pred = hypermodel.predict(x_space.reshape((-1, 1)))

fig = plt.figure()
plt.plot(x_space, y_f, c='b', label='f')
plt.plot(x_space, y_f_pred, c='orange', label='predicted f')

plt.scatter(x_train, y_train, c='b', label='training data')
plt.scatter(x_test, y_test, c='orange', label='test data')

plt.title(f"MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f}")
plt.legend()
plt.savefig('plot.pdf')

# Save model
file_name = "model.joblib"
joblib.dump(hypermodel, file_name)
