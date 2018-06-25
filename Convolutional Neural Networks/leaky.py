import cv2, os, pickle, numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten

import matplotlib.pyplot as plt

final = {}

def read_data(link):

    files = os.listdir(link)
    idx = 0
    for file1 in files:
        now = link + file1
        final[idx] = file1.split(".")[0]
        if idx == 0:
            train = np.load(now)
            m_lbl = np.array([idx] * train.shape[0])

        else:
            temp1 = np.load(now)
            temp3 = np.array([idx] * temp1.shape[0])
            train = np.vstack([train, temp1])
            m_lbl = np.hstack([m_lbl, temp3])

        idx += 1

    print(final)
    print(train.shape)
    return train, m_lbl

train, m_lbl = read_data("../col-774-spring-2018/train/")
test = np.load("../col-774-spring-2018/test/test.npy")

train_x, m_lbl = shuffle(train, m_lbl, random_state=0)
train_y = to_categorical(m_lbl, num_classes=20)

# train_x -= 255
# test -= 255

train_x = np.divide(train_x, 255)
test_x = np.divide(test, 255)

train_x.resize(train_x.shape[0], 28, 28, 1)
test_x.resize(test_x.shape[0], 28, 28, 1)

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(20, activation='softmax'))

fashion_model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
fashion_model.summary()
Y_train = np_utils.to_categorical(y_train, 20)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False) # randomly flip images

datagen.fit(train_x)

history = fashion_model.fit_generator(datagen.flow(train_x, train_y, batch_size=100),steps_per_epoch=1000, epochs=100)

predicted_classes = fashion_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

with open("submit_leaky.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(y_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[predicted_classes[i]] + '\n')

    print("done")


# fashion_model = Sequential()
# fashion_model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D((2, 2),padding='same'))
# fashion_model.add(Dropout(0.15))
# fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Dropout(0.15))
# fashion_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Dropout(0.3))
# fashion_model.add(Flatten())
# fashion_model.add(Dense(1024, activation='linear'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(Dropout(0.2))
# fashion_model.add(Dense(20, activation='softmax'))
