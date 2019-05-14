from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import PIL
import h5py
import sys

import pandas as pd
from keras.preprocessing.image import *
np.random.seed(2017)

def fun():

    X_train = []
    X_test = []
    for filenames in ["D:\software\PythonWorkspace\catdog\src\gap_wrapper.h5"]:
        filename = filenames
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    inputs = Input(X_train.shape[1:])#shape=（2048*3，）
    x = Dropout(0.5)(inputs)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)

    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2,verbose=2)

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("../img", (224,224), shuffle=False,
                                                 batch_size=1, class_mode=None)

    input_tensor = Input((224, 224, 3))
    x = input_tensor

    base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
    model2 = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    test = model2.predict_generator(test_generator, test_generator.samples)
    print(test)
    predict_y=model.predict(test,verbose=1)
    print(predict_y)

if __name__=="__main__":
    fun()