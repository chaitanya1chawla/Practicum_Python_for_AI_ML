import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, neighbors
from sklearn.model_selection import GridSearchCV, train_test_split


def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwopts):
    grid_search = GridSearchCV(base_model, paramgrid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(features, targets)
    return grid_search


def main():
    # import data
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # computing parameters
    h = .02  # step size in the mesh

    # Run SVM classifiers

    # KNN
    knn_params = {'n_neighbors': np.arange(10, 20),
                  'algorithm': ['ball_tree', 'kd_tree']}
    knn = neighbors.KNeighborsClassifier()
    knn_model = find_hyperparams(knn, knn_params, x_train, y_train)
    knn_train_sc = knn_model.score(x_train, y_train)
    knn_test_sc = knn_model.score(x_test, y_test)
    # SVM linear
    svc_params = {'kernel': ['linear'],
                  'C': [0.05, 0.1, 0.15]}
    svc = svm.SVC()
    svc_model = find_hyperparams(svc, svc_params, x_train, y_train)
    svc_train_sc = svc_model.score(x_train, y_train)
    svc_test_sc = svc_model.score(x_test, y_test)
    # SVM RBF
    rbf_params = {'kernel': ['rbf'],
                  'C': [0.9, 1]}
    rbf = svm.SVC()
    rbf_model = find_hyperparams(rbf, rbf_params, x_train, y_train)
    rbf_train_sc = rbf_model.score(x_train, y_train)
    rbf_test_sc = rbf_model.score(x_test, y_test)
    # SVM Poly
    poly_params = {'kernel': ['poly'],
                   'C': [0.4, 0.5, 0.6],
                   'coef0': [1],
                   'degree': [3, 5, 7]}
    poly = svm.SVC()
    poly_model = find_hyperparams(poly, poly_params, x_train, y_train)
    poly_train_sc = poly_model.score(x_train, y_train)
    poly_test_sc = poly_model.score(x_test, y_test)

    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = [f'KNN, train: {knn_train_sc:.2f}, test {knn_test_sc:.2f}\n{knn_model.best_params_}',
              f'linear SVC, train: {svc_train_sc:.2f}, test {svc_test_sc:.2f}\n{svc_model.best_params_}',
              f'poly SVC, train: {poly_train_sc:.2f}, test {poly_test_sc:.2f}\n{poly_model.best_params_}',
              f'rbf SVC, train: {rbf_train_sc:.2f}, test {rbf_test_sc:.2f}\n{rbf_model.best_params_}']

    for i, clf in enumerate((knn_model, svc_model, poly_model, rbf_model)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)

        # Plot also the training points
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.viridis, edgecolors='k')
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    # plt.show()
    figname = 'plot.pdf'
    plt.savefig(figname)


if __name__ == '__main__':
    main()
