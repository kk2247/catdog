import h5py
import numpy as np
from keras.layers import Dense,Input,Dropout
from keras.models import Model

import pandas as pd
from keras.preprocessing.image import *
np.random.seed(2017)

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

y_pred = model.predict(X_test, verbose=1)
print(y_pred.__len__())
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.read_csv("C:\\Users\\KGZ\\Desktop\\DeepLearningProject\\sample_submission.csv")

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("C:\\Users\\KGZ\\Desktop\\DeepLearningProject\\test", (224, 224), shuffle=False,
                                         batch_size=1, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('\\')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('pred.csv', index=None)
df.head(10)
