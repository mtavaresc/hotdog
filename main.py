from datetime import datetime

import tensorflowjs as tfjs
from keras import backend as k
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def get_model(input_shape):
    # Defining the architecture of our network
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    # Compiling the model and setting up the data augmentation
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    begin = datetime.now().replace(microsecond=0)

    # dimensions of our images.
    img_width, img_height = 150, 150

    train_data_dir = "data/train"
    validation_data_dir = "data/validation"
    nb_train_samples = 210
    nb_validation_samples = 100
    epochs = 100
    batch_size = 16

    if k.image_data_format() == "channels_first":
        inputs = (3, img_width, img_height)
    else:
        inputs = (img_width, img_height, 3)

    my_model = get_model(inputs)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # this is the augmentation configuration we will use for testing
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Creating the data generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="binary"
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="binary"
    )

    my_model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
    )

    # SavedModel.
    # The model architecture, and training configuration (including the optimizer, losses, and metrics)
    # are stored in `saved_model.pb`. The weights are saved in the `variables/` directory.
    my_model.save("model")

    # Keras also supports saving a single HDF5 file containing the model's architecture, weights values,
    # and `compile()` information. It is a light=weight alternative to SavedModel.
    my_model.save("model/ft.h5")

    # The TensorFlow.js converter has two components:
    # 1. A command line utility that converts Keras and TensorFlow models for use in TensorFlow.js.
    # `$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model`
    # 2. An API for loading and execute the model in the browser with TensorFlow.js.
    tfjs.converters.save_keras_model(my_model, "tfjs")

    end = datetime.now().replace(microsecond=0)
    print(f"\n\nElapsed time: {end - begin}")
