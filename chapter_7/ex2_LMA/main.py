import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
#from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Read text input
file_name = "breit_wigner.csv"
db = pd.read_csv(file_name, delimiter=',')

print(db.head(10))


# fit data to breit wigner function


def breit_wigner(x, a, b, c):
    y = a / (np.square(b - x[:, 0]) + c)
    return y


x = np.array([[0], [1], [4]])
params = [0.5, 0.2, 1]
y = breit_wigner(x, *params)


def eval_errors(x_all: np.array, y_all, func, params: list):
    return y_all - func(x_all, *list(params))


def eval_jacobian(x, func, params, h=0.0001):
    jac_cols = []
    for i in range(len(params)):
        dparams1 = params.copy()
        dparams2 = params.copy()
        dparams1[i] += h
        dparams2[i] -= h
        jac_col = (func(x, *dparams1) - func(x, *dparams2)) / 2*h
        jac_cols.append(jac_col)
    jac = np.vstack(jac_cols).T
    return jac




jac = eval_jacobian(np.array([[0], [1]]),
                         breit_wigner, [0.5, 0.2, 1])


def _lma_quality_measure(x, y_hat, func,
                                   params, delta_params, jac, lma_lambda):

    e = y_hat - func(x, *params)
    next_params = list(np.asarray(params) + np.asarray(delta_params))
    e_next = y_hat - func(x, *next_params)
    lma_rho = (np.dot(e.T, e) - np.dot(e_next.T, e_next)) / \
              delta_params.T @ (np.dot(lma_lambda, delta_params) + np.dot(jac.T, e))
    return lma_rho


def get_params(x, y_hat, func, params, jac, lma_lambda):
    # delta_params = np.zeros((3))
    e = y_hat - func(x, *params)
    # delta_params = np.dot(np.linalg.inv(np.dot(jac.T, jac)), np.dot(jac.T, e))
    delta_params = np.linalg.inv(np.matmul(jac.T, jac) + lma_lambda*np.eye(len(params))) \
                   @ np.dot(jac.T, e)
    return delta_params


lma_lambda = None
x = db[["x"]].to_numpy()
y = db["g"].to_numpy()
params = [60e3, 80, 800]


for i in range(10000):

    jac = eval_jacobian(x, breit_wigner, params)
    # calculate lambda if not set
    if lma_lambda is None:
        lma_lambda = np.linalg.norm(jac.T @ jac)
    e = eval_errors(x, y, breit_wigner, params)
    delta_params = get_params(x, y, breit_wigner, params, jac,lma_lambda)

    # calculate the quality measure
    lma_rho = _lma_quality_measure(x, y, breit_wigner,
                                       params, delta_params, jac, lma_lambda)
    if lma_rho > 0.75:
        lma_lambda /= 3
    elif lma_rho < 0.25:
        lma_lambda *= 2
    else:
        lma_lambda = lma_lambda
    # only change parameters if the quality measure is greater 0
    if lma_rho > 0:
        params = [x + d for x, d in zip(params,
                                            delta_params)]
    if i == 10000:
        break


print(params)


def lma(X_all, y_all, func, param_guess, **kwargs):
    fit = curve_fit(lambda x, *params: func(x.reshape(-1,1), *params), X_all.reshape(-1), y_all, param_guess)

    return fit[0]


x = db[["x"]].to_numpy()
y = db["g"].to_numpy()
fit = lma(x, y, breit_wigner,
          np.random.rand(3) * 1, )  # WILL NOT ALWAYS CONVERGE

print(fit)  # [a, b, c], fitting the function


# plot
fig = plt.figure()
plt.scatter(x, y, c='b')

x_plot = np.arange(0, 200, 0.1).reshape(-1,1)
bw_result = breit_wigner(x_plot, *fit)
plt.plot(x_plot, bw_result, c='b')
plt.show()

figname = 'plot.pdf'
plt.savefig(figname)
