import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.utils import shuffle
import os

final = {}

def read_data(link):

    files = os.listdir(link)
    idx = 0
    for file1 in files:
        now = link + file1
        final[idx] = file1.split(".")[0]
        if idx == 0:
            train = np.load(now)
            main = np.array([final[idx]] * train.shape[0])
            temp = np.zeros(20)
            temp[idx] = 1
            m_lbl = np.array([temp] * train.shape[0])

        else:
            temp1 = np.load(now)
            temp2 = np.array([final[idx]] * temp1.shape[0])
            temp = np.zeros(20)
            temp[idx] = 1
            temp3 = np.array([temp] * temp1.shape[0])
            train = np.vstack([train, temp1])
            main = np.hstack([main, temp2])
            m_lbl = np.vstack([m_lbl, temp3])

        idx += 1

    print(final)
    print(train.shape)
    return train, main, m_lbl


train, main, m_lbl = read_data("../col-774-spring-2018/train/")
test = np.load("../col-774-spring-2018/test/test.npy")

train, m_lbl = shuffle(train, m_lbl, random_state=0)

train = np.divide(train, np.max(train))
test = np.divide(test, np.max(test))

nodes = [32, 64, 128, 256, 512, 1024]  # number of nodes in the hidden layer
lrs = [0.1, 0.005, 0.001, 0.0001]  # learning rate, default = 0.001
epochs = 15
batch_size = 100

def build_model(nodes=10, lr=0.001):
    model = Sequential()
    model.add(Dense(nodes, kernel_initializer='uniform', input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('softmax'))

    # opt = optimizers.RMSprop(lr=lr)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)

model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch_size, verbose=0)

X = train
Y = m_lbl

param_grid = dict(nodes=nodes, lr=lrs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1, refit=True, verbose=2)

grid_result = grid.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

pred_classes = grid.predict(test)

with open("submit_NN.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(pred_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[pred_classes[i]]+ '\n')

    print("done")