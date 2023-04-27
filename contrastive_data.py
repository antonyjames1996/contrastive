import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

import matplotlib.pyplot as plt
from random_eraser import get_random_eraser
import tqdm

tf.debugging.set_log_device_placement(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

with tf.device("GPU:0"):
    os.system(
        '!wget --no-check-certificate \
        "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \
        -O "/tmp/cats-and-dogs.zip"'
    )
    # !wget --no-check-certificate \
    #     "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \
    #     -O "/tmp/cats-and-dogs.zip"

    # !wget  "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"

    local_zip = "/tmp/cats-and-dogs.zip"
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall("/tmp")
    zip_ref.close()

    try:
        os.makedirs("/tmp/cats-v-dogs")
        os.makedirs("/tmp/cats-v-dogs/training")
        os.makedirs("/tmp/cats-v-dogs/testing")
        os.makedirs("/tmp/cats-v-dogs/training/cats")
        os.makedirs("/tmp/cats-v-dogs/training/dogs")
        os.makedirs("/tmp/cats-v-dogs/testing/cats")
        os.makedirs("/tmp/cats-v-dogs/testing/dogs")
    except OSError:
        pass

    def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        files = []
        for filename in os.listdir(SOURCE):
            file = SOURCE + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")

        training_length = int(len(files) * SPLIT_SIZE)
        testing_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0:training_length]
        testing_set = shuffled_set[-testing_length:]

        for filename in training_set:
            this_file = SOURCE + filename
            destination = TRAINING + filename
            copyfile(this_file, destination)

        for filename in testing_set:
            this_file = SOURCE + filename
            destination = TESTING + filename
            copyfile(this_file, destination)

    CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
    TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
    TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

    split_size = 0.9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    # Expected output
    # 666.jpg is zero length, so ignoring
    # 11702.jpg is zero length, so ignoring

    TRAINING_DIR = "/tmp/cats-v-dogs/training/"
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, preprocessing_function=get_random_eraser(v_l=0, v_h=1)
    )
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR, batch_size=250, class_mode="binary", target_size=(64, 64)
    )

    VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, preprocessing_function=get_random_eraser(v_l=0, v_h=1)
    )
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, batch_size=250, class_mode="binary", target_size=(64, 64)
    )

    train_generator.reset()
    X_train, y_train = next(train_generator)
    batch_size = 250
    for i in tqdm.tqdm(range(int(train_generator.n / batch_size) - 1)):
        img, label = next(train_generator)
        X_train = np.append(X_train, img, axis=0)
        y_train = np.append(y_train, label, axis=0)
    print(X_train.shape, y_train.shape)

    validation_generator.reset()
    X_test, y_test = next(validation_generator)
    batch_size = 250
    for i in tqdm.tqdm(range(int(validation_generator.n / batch_size) - 1)):
        img, label = next(validation_generator)
        X_test = np.append(X_test, img, axis=0)
        y_test = np.append(y_test, label, axis=0)
    print(X_test.shape, y_test.shape)

    num_classes = 2
    input_shape = (64, 64, 3)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.02),
            layers.experimental.preprocessing.RandomWidth(0.2),
            layers.experimental.preprocessing.RandomHeight(0.2),
        ]
    )

    # Setting the state of the normalization layer.
    data_augmentation.layers[0].adapt(X_train)

    def create_encoder():
        resnet = keras.applications.ResNet50V2(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )

        inputs = keras.Input(shape=input_shape)
        # augmented = data_augmentation(inputs)
        # outputs = resnet(augmented)
        outputs = resnet(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name="cat-vs-dog-encoder")
        return model

    encoder = create_encoder()
    encoder.summary()

    learning_rate = 0.001
    batch_size = 265
    hidden_units = 512
    projection_units = 128
    num_epochs = 10
    dropout_rate = 0.5
    temperature = 0.05

    def create_classifier(encoder, trainable=True):
        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = keras.Input(shape=input_shape)
        features = encoder(inputs)
        features = layers.Dropout(dropout_rate)(features)
        features = layers.Dense(hidden_units, activation="relu")(features)
        features = layers.Dropout(dropout_rate)(features)
        outputs = layers.Dense(num_classes, activation="softmax")(features)

        model = keras.Model(
            inputs=inputs, outputs=outputs, name="cat-vs-dog-classifier"
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    class SupervisedContrastiveLoss(keras.losses.Loss):
        def __init__(self, temperature=1, name=None):
            super().__init__(name=name)
            self.temperature = temperature

        def __call__(self, labels, feature_vectors, sample_weight=None):
            # Normalize feature vectors
            feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
            # Compute logits
            logits = tf.divide(
                tf.matmul(
                    feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
                ),
                self.temperature,
            )
            return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

    def add_projection_head(encoder):
        inputs = keras.Input(shape=input_shape)
        features = encoder(inputs)
        outputs = layers.Dense(projection_units, activation="relu")(features)
        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="catsVsDogs-encoder_with_projection-head",
        )
        return model

    encoder = create_encoder()

    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
    )

    encoder_with_projection_head.summary()

    history = encoder_with_projection_head.fit(
        x=X_train, y=y_train, batch_size=batch_size, epochs=50
    )

    encoder_with_projection_head.save("models/contrastive+randErasing.h5")
