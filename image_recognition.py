# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

# Pillow
import PIL
from PIL import Image

# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def convert_img(path, maxsize):
	img = Image.open(path).convert('L')   # convert image to 8-bit grayscale
	# Make aspect ratio as 1:1, by applying image crop.
	# Please note, croping works for this data set, but in general one
	# needs to locate the subject and then crop or scale accordingly.
	WIDTH, HEIGHT = img.size
	if WIDTH != HEIGHT:
		m_min_d = min(WIDTH, HEIGHT)
		img = img.crop((0, 0, m_min_d, m_min_d))
        # Scale the image to the requested maxsize by Anti-alias sampling.
	img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
	return np.asarray(img)
		
def load_data(img_names, label_data, max_size):
	imgs = []
	labels = []
	for img_name in img_names:
		img = convert_img(img_name[0][0], max_size)
		imgs.append(img)
	for label in label_data:
		labels.append(int(label))
	return (np.asarray(imgs), np.asarray(labels))

def setup_model():
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(100, 100)),
		keras.layers.Dense(128, activation=tf.nn.sigmoid),
		keras.layers.Dense(16, activation=tf.nn.sigmoid),
		keras.layers.Dense(31, activation=tf.nn.softmax)
	])

	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	return model

def normalize(prediction, normalize_range):
	normalized_data = prediction * normalize_range
	return normalized_data
