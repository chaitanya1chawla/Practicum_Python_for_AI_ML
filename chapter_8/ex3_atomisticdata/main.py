from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotting(y_pred_test, y_act_test, y_pred_train, y_act_train, mse, r2, mae):
    fig = plt.figure()

    # first plot
    axes = fig.add_subplot(121)
    # secondary axis for R^2
    secax = axes.secondary_yaxis('right')
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    axes.set_title("Learning curves")
    axes.set_xlabel('fraction of training data used')
    axes.set_ylabel('MSE')
    secax.set_ylabel('R^2')
    axes.plot(x, mse, color='b')
    axes.plot(x, r2, color='r')

    # second plot
    ax = fig.add_subplot(122)
    ax.set_title(f'Model R^2:{r2[-1]:.3f}, MAE:{mae[-1]:.3f}')
    ax.set_xlabel('Calculated gap')
    ax.set_ylabel('Model gap')
    ax.scatter(y_act_test, y_pred_test, color='r', label='test data')
    ax.scatter(y_act_train, y_pred_train, color='b', label='training data')
    ax.legend(loc="upper left")
    ax.plot([-1,6],[-1,6])
    # plt.show()
    plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')


def krr(x, y):
    r2 = []
    mse = []
    mae = []

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

    model = KernelRidge(kernel = 'rbf')
    # for i in range(40,400,40):
    for i in range(10, 101, 10):
        krr_par = {'kernel': ["linear",'rbf'],
                   "alpha": list(np.logspace(-3, 2, 5)),
                   "gamma": list(np.logspace(-3,2,5))}

        hypermodel = GridSearchCV(model, krr_par, cv=10, scoring="r2")
        # taking fraction of data ==> 10%,20%,30% ...
        # x_train = x_train.iloc[1:i]
        # y_train = y_train.iloc[1:i]
        fraction = int(i * len(x_train) / 100)
        hypermodel.fit(x_train.iloc[:fraction], y_train.iloc[:fraction])

        y_pred = hypermodel.predict(x_test)
        y_pred_train = hypermodel.predict(x_train)  # predicted values for training data, needed for second plot

        r2.append(r2_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
    print(f"Best kernel: {hypermodel.best_params_}")
    print(f"R2: {r2}")
    # Save model
    file_name = "model.joblib"
    dump(hypermodel.best_estimator_, file_name)

    return y_pred, y_test, y_pred_train, y_train, mse, r2, mae


if __name__ == "__main__":
    cred = pd.read_csv("nitride_compounds.csv", sep=',')
    X = cred.drop(['Number', 'Nitride', 'PBE Eg (eV)', 'Formation energy per atom (eV)'], axis=1)
    X = X.drop(['HSE Eg (eV)', 'Band offset (eV)'], axis=1)
    y = cred[['HSE Eg (eV)', 'Band offset (eV)']]

    y_predicted, y_actual, y_pred_train, y_train, mse, r2, mae = krr(X, y)
    plotting(y_predicted, y_actual, y_pred_train, y_train, mse, r2, mae)
