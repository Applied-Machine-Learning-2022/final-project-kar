import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from PIL import ImageFile

TARGET_SIZE = (400, 400)
IMAGE_DIR = 'C:/Users/14794/Documents/NACME_capstone/train_1/train_1/'
META_DATA_PATH = "C:/Users/14794/Documents/NACME_capstone/all_data_info.csv/all_data_info.csv"
BATCH_SIZE = 8

def create_data():

    meta_data = pd.read_csv(META_DATA_PATH)

    # remove missing records
    meta_data = meta_data.dropna()
    meta_data = meta_data[['artist','date', 'genre', 'new_filename']]
    print(meta_data)

    labels = [x for x in meta_data["artist"].unique()]
    num_artists = len(labels)

    train_df, test_df = train_test_split(meta_data, random_state=42, test_size=0.15, shuffle=True)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    imageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255.,
        dtype=np.float32
    )

    train_generator = imageDataGenerator.flow_from_dataframe(
        train_df,
        directory=IMAGE_DIR,
        x_col='new_filename',
        y_col='artist',
        classes=labels,
        target_size=TARGET_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size = BATCH_SIZE
    )

    test_generator = imageDataGenerator.flow_from_dataframe(
        test_df,
        directory=IMAGE_DIR,
        x_col='new_filename',
        y_col='artist',
        classes=labels,
        target_size=TARGET_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE
    )

    return train_generator, test_generator, num_artists

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    train_generator, test_generator, num_classes = create_data()
    model = build_model(num_classes)
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.CategoricalCrossentropy(),
                  metrics = ['accuracy'])

    history = model.fit_generator(train_generator, epochs = 10)


