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

# def vgg(input_tensor):

#     def two_conv_pool(x, F1, F2, name):
#         x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

#         return x

#     def three_conv_pool(x, F1, F2, F3, name):
#         x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name))(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

#         return x

#     net = input_tensor

#     net = two_conv_pool(net, 64, 64, "block1")
#     net = two_conv_pool(net, 128, 128, "block2")
#     net = three_conv_pool(net, 256, 256, 256, "block3")
#     net = three_conv_pool(net, 512, 512, 512, "block4")

#     net = Flatten()(net)
#     net = Dense(512, activation='relu', name='fc')(net)
#     net = Dense(20, activation='softmax', name='predictions')(net)

#     return net

def vgg():

    def two_conv_pool(x, F1, F2, name, flag=1):
        if flag == 0:
            x.add(Conv2D(F1, (3, 3), activation='linear', padding='same', input_shape=(28,28,1), name='{}_conv1'.format(name)))

        else:
            x.add(Conv2D(F1, (3, 3), activation='linear', padding='same', name='{}_conv1'.format(name)))

        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.1))
        x.add(Conv2D(F2, (3, 3), activation='linear', padding='same', name='{}_conv2'.format(name)))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.1))
        x.add(MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name)))

        return x

    def three_conv_pool(x, F1, F2, F3, name):
        x.add(Conv2D(F1, (3, 3), activation='linear', padding='same', name='{}_conv1'.format(name)))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.1))
        x.add(Conv2D(F2, (3, 3), activation='linear', padding='same', name='{}_conv2'.format(name)))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.1))
        x.add(Conv2D(F3, (3, 3), activation='linear', padding='same', name='{}_conv3'.format(name)))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.1))
        x.add(MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name)))

        return x

    net = Sequential()

    net = two_conv_pool(net, 64, 64, "block1", 0)
    net = two_conv_pool(net, 128, 128, "block2")
    net = three_conv_pool(net, 256, 256, 256, "block3")
    net = three_conv_pool(net, 512, 512, 512, "block4")

    net.add(Flatten())
    net.add(Dense(512, activation='relu', name='fc'))
    net.add(Dense(20, activation='softmax', name='predictions'))

    return net

epoch = 15
batch_size = 100
learning_rate = 0.001

# X = Input(shape=[28, 28, 1])
# y = vgg(X)
# model = Model(X, y, "VGG")
# opt = optimizers.Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# model.summary()

# data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True, horizontal_flip=True)

# data_generator.fit(train_x)

model = vgg()
opt = optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# model.fit_generator(data_generator.flow(train_x, train_y, batch_size=100), steps_per_epoch=1000, epochs=50)

history = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_split=0.15, verbose=1)
model.save("vgg16.h5")

y_classes = model.predict(test_x)
y_classes = np.argmax(y_classes, axis=-1)
print(y_classes[:30])

with open("submit_VGG16.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(y_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[y_classes[i]] + '\n')

    print("done")
