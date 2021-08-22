import cv2
import csv
import numpy as np
from skimage.util import random_noise
from skimage import io, color, exposure, filters, img_as_ubyte
lines=[]
images=[]
measurements=[]
with open('VarunData/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    next(csvfile)
    for line in reader:
        lines.append(line)
        
for line in lines:
    src_path=line[0]
    filename=src_path.split('/')[-1]
    current_path='VarunData/IMG/'+filename
    image=cv2.imread(current_path)
    #     print(image.shape)
    images.append(image)    
    measurement=float(line[3])
    measurements.append(measurement)
    #Noise
    noiseImg = img_as_ubyte(random_noise(image, mode='gaussian'))
    images.append(noiseImg)
    measurements.append(measurement)

    #Left
    src_path=line[1]
    filename=src_path.split('/')[-1]
    current_path='VarunData/IMG/'+filename
    image=cv2.imread(current_path)
    images.append(image)
    measurement=float(line[3])+0.2
    measurements.append(min(measurement,1))
    
    #Right
    src_path=line[2]
    filename=src_path.split('/')[-1]
    current_path='VarunData/IMG/'+filename
    image=cv2.imread(current_path)
    images.append(image)
    measurement=float(line[3])-0.2
    measurements.append(max(measurement,-1))
    
    
augmented_images, augmented_measurements=[],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model
import matplotlib.pyplot as plt
import keras

model=Sequential()
model.add(Lambda(lambda x:x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Dropout(0.8))
model.add(Convolution2D(24,(5,5), strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5), strides=(2,2),activation='relu'))   
model.add(Convolution2D(48,(5,5), strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
adamOptimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mse' , optimizer=adamOptimizer)
history_object=model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=5, verbose=1)
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('loss.png')
model.save('model.h5')
