# Data organization and model libraries		
from sort_functions import retrieve_data, sort_data, crawl_for_images
from image_recognition import load_data, setup_model, normalize
from microimaging import grad_viewer

# Machine Learning libraries
import tensorflow as tf
from tensorflow import keras

# Statistical Method library
from Statistical-Methods import statistical_error

# Utility libraries
import numpy as np
import csv
import matplotlib.pyplot as plt

# Ignoring General TensorFlow Warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


# Prepping Data Outputs for Verifications

data_output = "data.csv"
# opening the file with w+ mode truncates the file
f = open(data_output, "w+")
f.writelines(["Model, Error (%)\n"])
f.close()

# Setting up data collection from CSV and images

file_name = "oasis_cross_sectional.csv"	
sort_name = "MMSE" # Either CDR or MMSE
mri_types = ["t88_gfc_tra", "t88_masked_gfc_tra", "t88_masked_gfc_fseg_tra","t88_gfc_cor","111_t88_gfc_sag","111_sag_88","mpr-1_anon_sag_66","mpr-2_anon_sag_66","mpr-3_anon_sag_66","mpr-4_anon_sag_66"]

sort_dict = {
		"MMSE": 0,
		"CDR": 1
}

sort_number = sort_dict[sort_name.upper()] # Transforms sort name to 0 or 1


max_size = 100, 100 # Max Image Size

# Machine Learning Variables
total_epochs = 100
total_batch_size = 5
ML = True # Do Machine Learning (True/False)
OVERWRITE_MODELS = False # Overwrite Existing Models (True/False)


'''
Data Organization
'''

# Retrieving patient information
patient_data = retrieve_data(file_name)

total_patients = len(patient_data)

# Sorts data for closer inspection if need be
#
# sort_data(patient_data, sort_number)

# Start parsing mri types

for mri_type in mri_types:
	img_names, label_data = crawl_for_images(mri_type, patient_data, sort_number)

	model_name = mri_type + "_model.h5"
	
	save_location = "models/" + model_name
	
	if ML:
		(images, labels) = load_data(img_names, label_data, max_size)
	
		# Max (max of array)/ middle (middle of array)
		max = int(images.shape[0]-1)
		middle = int(images.shape[0]/2)

		# Uses middle of arrays to divide images and labels for training and testing
		train_images = images[0:middle]
		train_labels = labels[0:middle]
		test_images = images[int(middle+1):max]
		test_labels = labels[int(middle+1):max]

		# Normalizes the values from (0 to 1)
		train_images = train_images / 255.0
		test_images = test_images / 255.0

		path_exists = os.path.exists(save_location)

		# Checks if model exists and that model doesn't need retraining
		if(path_exists and not OVERWRITE_MODELS):
			print("Loading existing model for " + mri_type)
			model = keras.models.load_model(save_location)
		else:
			# Begin Modelling
			model = setup_model()

			# Fits model for image_recognition
			model.fit(train_images, train_labels, validation_data=(test_images,test_labels), epochs = total_epochs, batch_size = total_batch_size, verbose = 2)

			# Saves model
			keras.models.save_model(model, save_location)
		
		# Uses test images to predict outcomes
		predictions = model.predict(test_images)
		normalized_predictions = []
		normalize_range = 30
		counter = 0
		correct_predictions = 0
	
		errors = []

		# Organizes the predictions

		for prediction in predictions:
			max = np.amax(prediction)
			outcomes = np.where(prediction == max)[0][0]
			if not(isinstance(outcomes, list)):
				outcomes = [outcomes]
			for outcome in outcomes:
				# Actual value
				actual = test_labels[counter]
				
				# Predicted value
				predicted = outcome

				# Estimates error using actual value and predicted value
				error = statistical_error(actual, predicted)
			
				errors.append(error)
				counter += 1
	
	# Creates rows of mri type name and error 
	row = [mri_type,str((np.average(errors)*100))]
	
	with open(data_output, 'a') as csvFile:
		file = csv.writer(csvFile)
		file.writerow(row)