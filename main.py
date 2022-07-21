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
    # new_df = new_df.dropna()

    # x_train, x_test, y_train, y_test = train_test_split(FEATURES, TARGET, test_size = 0.15, random_state = 42, shuffle =True)
    # print( x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # You can see after the 90th image it gives you an error IOError: image file truncated (80 bytes not processed)
    # This happens when the file is too big. Lets print out the image
    # print(list_dir[90])
    # This prints out image 100102.jpg

    # To fix this jsut add this statement above your call stack
    ImageFile.LOAD_TRUNCATED_IMAGES = True


    # Turning it into a byte array
    painting_images = []
    for i in tqdm(range(list_df.shape[0])):
        img = image.load_img('train_1/train_1/'+ new_df['new_filename'][i], target_size=(100, 100, 3))
        img = image.img_to_array(img)
        # img = keras.preprocessing.image.smart_resize(img, (100, 100))
        img = img/255
        painting_images.append(img)
        # print('train_1/train_1/'+ new_df['new_filename'][i])
        # print(img)
    X = np.array(painting_images)

    # print(plt.imshow(X[2]))


     # print(new_df.columns)
    # print(new_df[0])

    # FEATURES = [new_df[['date', 'genre','style' ]], X]
    # # print(FEATURES)
    # TARGET = new_df[['artist']]
    # # print(TARGET)

    # YYYYYYYYYEEEETTTTT
    y = np.array(new_df['artist'])
    # y = np.array(new_df['date', 'genre', 'style'], X)

    # x = []
    # for a, b in new_df.iterrows():
    #     array = []
    #     for paintings in np.nditer(X):
    #         p = X(paintings)
    #
    #         d = str(b['date'])
    #         g = str(b['genre'])
    #         s = str(b['style'])
    #         array.append(d)
    #         array.append(g)
    #         array.append(s)
    #         array.append(p)
    #     print(array)
    #     x.append(array)

    labels = [x for x in new_df["artist"].unique()]
    num_artists = len(labels)

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42, shuffle =True)

    return X, y, num_artists

    # return x_train, x_test, y_train, y_test, num_artists

def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(100,100,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
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
if __name__ == "__main__":

    x_array, y_array, num_classes = data_preprocessing()
    print(x_array.shape)
    print(y_array.shape)
    # dataframe = pd.DataFrame(x_array.reshape((y_array.shape[0], -1)), y_array, columns=['image_bytes', 'filename'])
    dataframe_1 = pd.DataFrame(x_array.reshape((y_array.shape[0], -1)), columns=list(range(30000)))
    dataframe_2 = pd.DataFrame(y_array, columns=["filename"])
    dataframe = pd.merge(dataframe_1, dataframe_2, left_index=True, right_index=True)
    dataframe = dataframe.dropna()

    # labels = [x for x in dataframe["artist"].unique()]

    print(dataframe.isna().sum())

    X_array = dataframe[list(range(30000))].to_numpy()
    Y_array = dataframe['filename'].to_numpy()
    artists = list(np.unique(Y_array))
    num_classes = len(artists)
    Y_array = np.array([artists.index(i) for i in Y_array])
    Y_array = to_categorical(Y_array)

    # for painters in y_array:
    #     if painters = naN


    # print(x_array[0])
    # print(y_array[0])

    x_train, x_test, y_train, y_test = train_test_split(X_array, Y_array, random_state = 42, test_size = 0.15, shuffle = True)
    x_train = x_train.reshape((y_train.shape[0], 100, 100, 3))
    x_test = x_test.reshape((y_test.shape[0], 100, 100, 3))
    # print(x_train, y_test, y_train, y_test)
    print(x_train.shape, y_test.shape, y_train.shape, y_test.shape)

    # x_train, x_test, y_train, y_test, num_classes = data_preprocessing()
    model = create_model(num_classes)
    model.summary()
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
    #
    history = model.fit(x_train, y_train, epochs=10, batch_size = 24, validation_split = 0.15)
    model.save('./ckpts/epoch10')
    #
    # print(x_train.shape)
    #
    # model.fit(x_train, y_train, epochs = 10, batch_size= 64)

