import numpy as np
from keras.models import Sequential, Model, Input
from keras.models import load_model
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, Average
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import os
import keras.backend as k

# Striving for Simplicity: The All Convolutional Net
def conv_pool_cnn(model_input):

    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(20, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='conv_pool_cnn')

    return model

# Striving for Simplicity: The All Convolutional Net
def all_cnn(model_input):

    x = Conv2D(
        96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(20, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='all_cnn')

    return model

# Striving for Simplicity: The All Convolutional Net
def nin_cnn(model_input):

    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)

    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(20, (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='nin_cnn')

    return model

# https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f
def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models]

    # averaging outputs
    yAvg=Average()(yModels)

    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

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

model_input = Input(shape=(28, 28, 1))

#	c_model = conv_pool_cnn(model_input)
#	all_cnn_model = all_cnn(model_input)

conv_pool_model = load_model('conv_pool.h5')
all_cnn_model = load_model('all_cnn.h5')
vgg_aug_model = load_model('vgg_aug.h5')
vgg_model = load_model('vgg.h5')

models=[]

models.append(conv_pool_model)
models.append(all_cnn_model)
models.append(vgg_aug_model)
models.append(vgg_model)

model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
modelEns = ensembleModels(models, model_input)
modelEns.summary()


pred = modelEns.predict(train_x, verbose=1)
pred = np.argmax(pred, axis=1)
error = np.sum(np.not_equal(pred, train_y))
print(error)

predicted_classes = modelEns.predict(test_x, verbose=1)
predicted_classes = np.argmax(predicted_classes, axis=-1)
with open("submit_fire.csv", "w") as outfile:
    outfile.writelines("ID,CATEGORY\n")
    for i in range(predicted_classes.shape[0]):
        outfile.writelines(str(i) + ',' + final[predicted_classes[i]] + '\n')

    print("done")
