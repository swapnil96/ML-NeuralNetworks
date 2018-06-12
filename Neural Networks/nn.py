import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
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
            m_lbl = np.array([idx] * train.shape[0])

        else:
            temp1 = np.load(now)
            temp2 = np.array([final[idx]] * temp1.shape[0])
            temp3 = np.array([idx] * temp1.shape[0])
            train = np.vstack([train, temp1])
            main = np.hstack([main, temp2])
            m_lbl = np.hstack([m_lbl, temp3])

        idx += 1

    print(final)
    print(train.shape)
    return train, main, m_lbl

train, main, m_lbl = read_data("../col-774-spring-2018/train/")
test = np.load("../col-774-spring-2018/test/test.npy")

train_x, m_lbl = shuffle(train, m_lbl, random_state=0)
train_y = to_categorical(m_lbl, num_classes=20)

train_x = np.divide(train_x, np.max(train_x))
test_x = np.divide(test, np.max(test))

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, train_size=0.8)

nodes = [32, 64, 128, 256, 512, 1024]  # number of nodes in the hidden layer
lrs = [0.1, 0.01, 0.001, 0.0001]  # learning rate, default = 0.001
epoch = 150
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

# for node in nodes:

#     for lr in lrs:

#         mod = build_model(node, lr)
#         history = mod.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(valid_x, valid_y), verbose=1)

#         # print(history.history.keys())
#         fig1 = plt.figure()
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('model accuracy')
#         plt.ylabel('accuracy')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'valid'], loc='upper left')
#         # plt.show()
#         fig1.savefig('acc_'+str(node)+'_'+str(lr)+'.png')
#         # summarize history for loss
#         fig2 = plt.figure()
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'valid'], loc='upper left')
#         # plt.show()
#         fig2.savefig('loss_'+str(node)+'_'+str(lr)+'.png')
#         # break

#     # break
#     print("Done for", node, "nodes")

model = build_model(2048, 0.001)
model.fit(train_x, train_y, epochs=15, batch_size=batch_size, validation_data=(valid_x, valid_y), verbose=1)
y_prob = model.predict(test_x)
y_classes = y_prob.argmax(axis=-1)

with open("submit_NN_2048.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(y_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[y_classes[i]] + '\n')

    print("done")