import numpy as np
import sklearn
import keras
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

import loadData


def get_test_data():
    """
    Taken from http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
                        #sphx-glr-auto-examples-linear-model-plot-ols-py
    """
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    y = diabetes.target
    # Split the data into training/testing sets
    X_train = diabetes_X[:-20]
    X_test = diabetes_X[-20:]
    # Split the targets into training/testing sets
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]
    return X_train, y_train, X_test, y_test


def sklearn_model_test(X_train, y_train, X_test, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))


def get_data(tsize):
    """
    load in data
    use train_test_split
    #TODO does preprocessing data or scaling data help when training model, please discuss.
    :return: two numpy arrays containing data (RDF,Spec)
    """
    # TODO xuruiqin

    X, y = loadData.rawData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, shuffle=True)
    return X_train, y_train, X_test, y_test


def sklearn_model(X_train, y_train, X_test, y_test):
    """
    TODO build a model, and investigate the hyperparameters
    """

    # regr = linear_model.LinearRegression()
    # regr.fit(X_train, y_train)
    # y_pred = regr.predict(X_test)
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(y_test, y_pred))
    # print('Variance score: %.2f' % r2_score(y_test, y_pred))

    reg1 = linear_model.Lasso(alpha=0.1)
    reg1.fit(X_train, y_train)
    y_pred = reg1.predict(X_test)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Variance score: %.2f" % r2_score(y_test, y_pred))


def keras_model(X_train, y_train, X_test, y_test):
    """
    TODO build a model, and investigate the hyperparameters
    """
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=128))
    model.add(Dense(units=129, activation='softmax'))
    model.compile(loss='mse',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
    print loss_and_metrics


def learn(X_train, y_train, X_test, y_test):
    sklearn_model(X_train, y_train, X_test, y_test)
    keras_model(X_train, y_train, X_test, y_test)


def results():
    """
    TODO print out results and discuss them
    """
    pass


print " IR project using ML and DL."

"""
X_train,y_train,X_test,y_test = get_test_data()
sklearn_model_test(X_train,y_train,X_test,y_test)
"""
for i in range(1, 100, 1):
    tsize = i/100.0
    X_train, y_train, X_test, y_test = get_data(tsize)
    sklearn_model(X_train, y_train, X_test, y_test)
    print tsize, "\n"
# learn(X_train, y_train, X_test, y_test)

# TODO
# results
