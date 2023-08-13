import numpy as np
np.random.seed(1000)
from tensorflow import keras
import cv2
import os
from PIL import Image

os.environ['KERAS_BACKEND']='tensorflow'


path="Lumpy Skin Images Dataset-1/";

# dataset=[]
# label=[]

# infected_images=os.listdir(path+'Lumpy Skin/')
infected_images=os.listdir(path+'Lumpy Skin/')


path = "Lumpy Skin images Dataset-1/Lumpy Skin/"
path="Lumpy Skin Images Dataset-1\Lumpy Skin"
SIZE = 256  # Specify the desired size for resizing the images
dataset = []
label = []

infected_images = os.listdir(path)
for i, image_name in enumerate(infected_images):
    image_path = os.path.join(path, image_name)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((SIZE, SIZE))
    dataset.append(np.array(image))
    label.append(0)


path2="Lumpy Skin Images Dataset-1/Normal Skin"
uninfected_images = os.listdir(path2)
for i, image_name in enumerate(uninfected_images):
    image_path = os.path.join(path2, image_name)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((SIZE, SIZE))
    dataset.append(np.array(image))
    label.append(1)

##### NETWORK ###

INPUT_SHAPE=(256,256,3)
inp=K=keras.layers.Input(shape=INPUT_SHAPE)

conv1=keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(inp)

pool1=keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
norm1=keras.layers.BatchNormalization(axis=-1)(pool1)
drop1=keras.layers.Dropout(rate=0.2)(norm1)

conv2=keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(drop1)
pool2=keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
norm2=keras.layers.BatchNormalization(axis=-1)(pool2)
drop2=keras.layers.Dropout(rate=0.2)(norm2)


flat=keras.layers.Flatten()(drop2)

hidden1=keras.layers.Dense(512,activation='relu')(flat)
norm3=keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3=keras.layers.Dropout(rate=0.2)(norm3)

hidden2=keras.layers.Dense(256,activation='relu')(drop3)
norm4=keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4=keras.layers.Dropout(rate=0.2)(norm4)

out=keras.layers.Dense(2,activation='sigmoid')(drop4)

model=keras.Model(inputs=inp,outputs=out)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

x_train, x_test, y_train, y_test=train_test_split(dataset, to_categorical(np.array(label)),test_size=0.20,random_state=0)

history=model.fit(np.array(x_train),y_train, batch_size=50, verbose=1, epochs=25, validation_split=0.1, shuffle=False)

print("test_Accuracu: {:.2f}%".format(model.evaluate(np.array(x_test),np.array(y_test))[1]*100))

model.save('lumpy_cnn.h5')