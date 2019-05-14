from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import PIL
import h5py

def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("C:\\Users\\KGZ\\Desktop\\DeepLearningProject\\train", image_size, shuffle=False,
                                              batch_size=1)
    test_generator = gen.flow_from_directory("C:\\Users\\KGZ\\Desktop\\DeepLearningProject\\test", image_size, shuffle=False,
                                             batch_size=1, class_mode=None)
    print(test_generator.__len__())
    print(train_generator.__len__())
    train = model.predict_generator(train_generator, train_generator.samples)
    test = model.predict_generator(test_generator, test_generator.samples)
    with h5py.File("gap_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

if __name__=="__main__":

    write_gap(ResNet50, (224, 224))
    # write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
    # write_gap(Xception, (299, 299), xception.preprocess_input)
