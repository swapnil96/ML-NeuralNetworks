import os, pickle, numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, Model, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt

final = {}

def read_data(link):

    files = os.listdir(link)
    files.sort()
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


train, m_lbl = read_data("../../col-774-spring-2018/train/")
test = np.load("../../col-774-spring-2018/test/test.npy")

train_x, m_lbl = shuffle(train, m_lbl, random_state=0)
train_y = to_categorical(m_lbl, num_classes=20)

# train_x -= 255
# test -= 255

train_x = np.divide(train_x, 255)
test_x = np.divide(test, 255)

train_x.resize(train_x.shape[0], 28, 28, 1)
test_x.resize(test_x.shape[0], 28, 28, 1)


def nin_cnn(model_input):

    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(20, (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='nin_cnn')

    return model

model = nin_cnn(Input(shape=(28,28,1)))

data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

data_generator.fit(train_x)

opt = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

model.fit_generator(data_generator.flow(train_x, train_y, batch_size=100), steps_per_epoch=1000, epochs=50)

# Save whole model to HDF5
model.save("model_nin_cnn.h5")
print("Saved model to disk")

y_classes = model.predict_classes(test_x)
print(y_classes[:30])

with open("submit_nin_cnn.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(y_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[y_classes[i]] + '\n')

    print("done")
