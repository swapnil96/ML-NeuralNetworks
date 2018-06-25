from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import cv2, numpy as np
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
            temp = np.zeros(20)
            temp[idx] = 1
            m_lbl = np.array([temp] * train.shape[0])
            # m_lbl = np.array([idx] * train.shape[0])

        else:
            temp1 = np.load(now)
            temp2 = np.array([final[idx]] * temp1.shape[0])
            temp = np.zeros(20)
            temp[idx] = 1
            temp3 = np.array([temp] * temp1.shape[0])
            # temp3 = np.array([idx] * temp1.shape[0])
            train = np.vstack([train, temp1])
            m_lbl = np.vstack([m_lbl, temp3])
            # m_lbl = np.hstack([m_lbl, temp3])

        idx += 1

    print(final)
    print(train.shape)
    return train, m_lbl

test = np.load("../col-774-spring-2018/test/test.npy")
train, m_lbl = read_data("../col-774-spring-2018/train/")

train.resize(train.shape[0], 28, 28, 1)
test.resize(test.shape[0], 28, 28, 1)

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

num_epochs = 5
batch_size = 100
learning_rate = 0.01
steps = 10

opt = optimizers.Adam(lr=learning_rate)
model = VGG_16()
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(train, m_lbl, batch_size=batch_size, epochs=num_epochs, verbose=1)

prediction = model.predict(test)

with open("submit_VGG1.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(prediction.shape[0]):
        outfile.writelines(str(i) + ',' + final[prediction[i]] + '\n')

print("done")