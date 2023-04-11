"""
Submission done for ex.1
"""


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets, neighbors
# from sklearn.model_selection import GridSearchCV, train_test_split
#
#
# def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwopts):
#     grid_search = GridSearchCV(base_model, paramgrid, cv=cv, scoring="accuracy", n_jobs=-1)
#     grid_search.fit(features, targets)
#     return grid_search
#
#
# def main():
#     # import data
#     iris = datasets.load_iris()
#     x = iris.data[:, :2]
#     y = iris.target
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#
#
# #nn.CrossEntropyLoss