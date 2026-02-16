import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pickle
from glob import glob
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

K.set_image_data_format('channels_last')


def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape


def get_num_of_classes():
	return len(glob('gestures/*'))


image_x, image_y = get_image_size()


def cnn_model(num_of_classes):
	model = Sequential()
	model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))

	sgd = optimizers.SGD(learning_rate=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	filepath = 'cnn_model.keras'
	checkpoint1 = ModelCheckpoint(
		filepath,
		monitor='val_accuracy',
		verbose=1,
		save_best_only=True,
		mode='max',
	)
	callbacks_list = [checkpoint1]
	return model, callbacks_list


def train():
	with open('train_images', 'rb') as f:
		train_images = np.array(pickle.load(f), dtype=np.float32)
	with open('train_labels', 'rb') as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open('val_images', 'rb') as f:
		val_images = np.array(pickle.load(f), dtype=np.float32)
	with open('val_labels', 'rb') as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1)) / 255.0
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1)) / 255.0

	num_of_classes = max(int(np.max(train_labels)), int(np.max(val_labels))) + 1
	num_of_classes = max(num_of_classes, get_num_of_classes())
	train_labels = to_categorical(train_labels, num_classes=num_of_classes)
	val_labels = to_categorical(val_labels, num_classes=num_of_classes)

	print(f'train_images: {train_images.shape}, val_images: {val_images.shape}')
	print(f'train_labels: {train_labels.shape}, val_labels: {val_labels.shape}')

	model, callbacks_list = cnn_model(num_of_classes)
	model.summary()

	batch_size = min(32, len(train_images))
	model.fit(
		train_images,
		train_labels,
		validation_data=(val_images, val_labels),
		epochs=15,
		batch_size=batch_size,
		callbacks=callbacks_list,
	)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print('CNN Error: %.2f%%' % (100 - scores[1] * 100))


train()
K.clear_session()
