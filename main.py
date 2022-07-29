import tensorflow.python.keras.preprocessing.image
from PIL import Image
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import tree
import io
import pydotplus

from IPython.display import Image


ImageDir = 'C:/Users/14794/Documents/NACME_capstone/train_1/train_1/'
# ending = '.jpg'
CSVDir = 'C:/Users/14794/Documents/NACME_capstone/all_data_info.csv/all_data_info.csv'

def data_preprocessing():

    # Load all_data_info.csv
    labels = pd.read_csv(CSVDir)

    # Creating dataframe from Image directory
    list_dir = os.listdir('train_1/train_1/')
    list_df = pd.DataFrame(list_dir, columns = ['new_filename'])
    for i in list_df['new_filename']:
        list_df['path'] = ImageDir + i
    # List df has 11025 rows

    # Joining dataframes drop non matching
    new_df = pd.merge(labels, list_df, on ="new_filename")
    ImageFile.LOAD_TRUNCATED_IMAGES = True


    # Turning it into a byte array
    painting_images = []
    for i in tqdm(range(list_df.shape[0])):
        img = image.load_img('train_1/train_1/'+ new_df['new_filename'][i], target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img/255
        painting_images.append(img)
        # print('train_1/train_1/'+ new_df['new_filename'][i])
        # print(img)
    X = np.array(painting_images)

    # Looking at other features
    d = []
    for a, b in new_df.iterrows():
        array = []
        g = str(b['genre'])
        s = str(b['style'])
        array.append(g)
        array.append(s)
        print(array)
        d.append(array)

    y = np.array(new_df['artist'])



    return X, y, d



def cleaning_data(x_array, y_array, features):

    # Create dataframe
    dataframe_1 = pd.DataFrame(x_array.reshape((y_array.shape[0], -1)), columns=list(range(150528)))
    dataframe_2 = pd.DataFrame(y_array, columns=["filename"])
    dataframe_3 = pd.DataFrame(features, columns= ["genre", "style"])
    df = pd.merge(dataframe_1, dataframe_3, left_index=True, right_index=True)
    df = pd.merge(df, dataframe_2, left_index = True, right_index = True)
    df = df.dropna()


    # drop columns where there are less than 10 paintings
    dataframe = df[df['count'] >= 10]
    print(dataframe.head(10))

    # see how many artists
    labels = [x for x in dataframe["filename"].unique()]
    num_artists = len(labels)
    print(num_artists)

    # turn back into arrays
    X_array = dataframe.iloc[:,:150528].to_numpy()
    Y_array = dataframe['filename'].to_numpy()
    artists = list(np.unique(Y_array))
    num_classes = len(artists)
    Y_array = np.array([artists.index(i) for i in Y_array])
    Y_array = to_categorical(Y_array)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X_array, Y_array, random_state=42, test_size=0.15, shuffle=True)
    print(x_train.shape, y_train.shape, y_train.shape, y_test.shape)


    # reshape to correct dimensions
    x_train = x_train.reshape((y_train.shape[0], 224, 224, 3))
    x_test = x_test.reshape((y_test.shape[0], 224, 224, 3))

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return x_train, x_test, y_train, y_test, num_artists, dataframe


# CNN method
def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5),  activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model

# RESNET50 method
def RESNET50(num_classes):
    resnet = Sequential()

    pretrained = tensorflow.keras.applications.ResNet50(include_top = False,
                                                        input_shape = (100, 100, 3),
                                                        pooling = 'avg',
                                                        classes = 5,
                                                        weights = 'imagenet')
    for layer in pretrained.layers:
        layer.trainable = False

    resnet.add(pretrained)
    resnet.add(Flatten())
    resnet.add(Dense(128, activation='relu'))
    resnet.add(Dropout(0.5))
    resnet.add(Dense(64, activation='relu'))

    resnet.add(Dense(32, activation='relu'))
    resnet.add(Dropout(0.5))


    resnet.add(Dropout(0.25))
    resnet.add(Dense(num_classes, activation='softmax'))

    return resnet



if __name__ == "__main__":


    x_array, y_array, feature_array = data_preprocessing()
    x_train, x_test, y_train, y_test, num_classes, data = cleaning_data(x_array, y_array, feature_array)
    print(data)
    print(data.columns)

    # Run CNN model
    model = create_model(num_classes)
    model.summary()
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
    history = model.fit(x_train, y_train, epochs=250, batch_size = 18)

    # Plot CNN model
    epochs = range(250)
    plt.plot(epochs, history.history['accuracy'])
    plt.show()
    plt.plot(epochs, history.history['loss'])
    plt.show()



    model.save('./ckpts/Model_1')

    evaluate = model.evaluate(x_test, y_test)
    print(evaluate)

    # Run RESNET50
    ResNet = RESNET50(num_classes)
    ResNet.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
    ResHistory = ResNet.fit(x_train, y_train, epochs = 100, batch_size = 15)
    ResNet.save('.ckpts/ResNet')

    # Plot RESNET
    ResNet.evaluate(x_test, y_test)
    plt.plot(epochs, ResHistory.history['accuracy'])
    plt.show()
    plt.plot(epochs, ResHistory.history['loss'])
    plt.show()



