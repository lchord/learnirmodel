# -*- coding: utf-8 -*-

import numpy as np
import sklearn
import keras
from sklearn import datasets
from sklearn import datasets, linear_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras import optimizers

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


def get_data():
    """
    load in data
    use train_test_split
    #TODO does preprocessing data or scaling data help when training model, please discuss.
    :return: two numpy arrays containing data (RDF,Spec)
    """
    # TODO xuruiqin

    X, y = loadData.rawData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=32)
    return X_train, y_train, X_test, y_test


def sklearn_model(X_train, y_train, X_test, y_test):
    """
    TODO build a model, and investigate the hyperparameters
    """
    reg1 = ExtraTreesRegressor(n_estimators=32, random_state=0)
    reg1.fit(X_train, y_train)
    y_pred = reg1.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred)
    r2s_sklearn = r2_score(y_test, y_pred)
    return mse_sklearn, r2s_sklearn


def keras_model(X_train, y_train, X_test, y_test):
    """
    TODO build a model, and investigate the hyperparameters
    """
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=128))
    model.add(Dense(units=129, activation='relu'))
    model.compile(loss='mse',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=128)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    mse_keras = loss_and_metrics[0]
    acc_keras = loss_and_metrics[1]
    return mse_keras, acc_keras


def learn(X_train, y_train, X_test, y_test):
    sklearn_model(X_train, y_train, X_test, y_test)
    keras_model(X_train, y_train, X_test, y_test)


def results(X_train, y_train, X_test, y_test):
    """
    TODO print out results and discuss them
    """
    mse_sklearn, r2s_sklearn = sklearn_model(X_train, y_train, X_test, y_test)
    mse_keras, acc_keras = keras_model(X_train, y_train, X_test, y_test)
    # Results
    print "\nResults:"
    print "Sklearn Model    Mean Squared Error: %.2f    R2_Score: %.2f" % (mse_sklearn, r2s_sklearn)
    print "Keras Model      Mean Squared Error: %.2f    Accuracy: %.2f" % (mse_keras, acc_keras)
    # Discussion
    print "\nDiscussion:"
    print "Here we've got the mean squared errors(MSEs) of two models."
    if mse_sklearn < mse_keras:
        print "The MSE of Sklearn model is smaller.\nSklearn model is better."
    else:
        print "The MSE of Keras model is smaller.\nKeras model is better."


print "IR project using ML and DL.\nPlease wait..."

# X_train,y_train,X_test,y_test = get_test_data()
# sklearn_model_test(X_train,y_train,X_test,y_test)

X_train, y_train, X_test, y_test = get_data()

# TODO
# results
results(X_train, y_train, X_test, y_test)
