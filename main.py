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


path = r'C:/Users/14794/Documents/NACME_capstone/train_1/train_1/'
ending = '.jpg'

# Load all_data_info.csv
labels = pd.read_csv('C:/Users/14794/Documents/NACME_capstone/all_data_info.csv/all_data_info.csv')
# print(labels)

# Creating calling path
list_dir = os.listdir('train_1/train_1/')
list_df = pd.DataFrame(list_dir, columns = ['new_filename'])
for i in list_df['new_filename']:
    list_df['path'] = path + i
# print(list_df)
# List df has 11025 rows
#
# Joining dataframes drop non matching
new_df = pd.merge(labels, list_df, on ="new_filename")
# print(new_df.head(10))
# new_df = new_df.dropna()
# print(new_df)

    # print(b['date'])
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
    img = image.load_img('train_1/train_1/'+ new_df['new_filename'][i], target_size = (400,400,3))
    img = image.img_to_array(img)
    img = img/255
    painting_images.append(img)
    # print('train_1/train_1/'+ new_df['new_filename'][i])
    # print(img)
X = np.array(painting_images)

print(plt.imshow(X[2]))

# imageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator()
#
# trainGenerator = imageDataGenerator.flow_from_dataframe(new_df, x_col="new_filename", y_col="artist")



 # print(new_df.columns)


# FEATURES = [new_df[['date', 'genre','style' ]], X]
# # print(FEATURES)
# TARGET = new_df[['artist']]
# # print(TARGET)

# YYYYYYYYYEEEETTTTT
y = np.array(new_df['artist'])
# y = np.array(new_df['date', 'genre', 'style'], X)

# y = []
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
#     y.append(array)
#
#
# print(x)
# print()
# print(y)
# print()
# print(x.shape, y.shape)

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
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
model.add(Dense(25, activation='sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'catergorical_crossentropy', metrics = ['accuracy'])


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.15, shuffle = True)

model.fit(x_train, y_train, epochs = 10, batch_size= 64)
