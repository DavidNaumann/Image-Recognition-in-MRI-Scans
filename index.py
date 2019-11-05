'''
[['patient no. 1', [mmse, cdr]],['patient no. 2', [mmse, cdr]], etc. ]
MMSE = Mini-Mental State Exam
CDR = Clinical Dementia Rating
'''
		
from sort_functions import retrieve_data, sort_data, crawl_for_images
from image_recognition import load_data, setup_model, normalize

import tensorflow as tf
from tensorflow import keras

import numpy as np

file_name = "oasis_cross_sectional.csv"	
sort_name = "MMSE" # Either CDR or MMSE
mri_types = ["t88_gfc_tra", "t88_masked_gfc_tra", "t88_masked_gfc_fseg_tra","t88_gfc_cor","111_t88_gfc_sag","111_sag_88","mpr-1_anon_sag_66","mpr-2_anon_sag_66","mpr-3_anon_sag_66","mpr-4_anon_sag_66"]

max_size = 100, 100

total_epochs = 500
total_batch_size = 10

ML = True

sort_dict = {
		"MMSE": 0,
		"CDR": 1
}

sort_number = sort_dict[sort_name.upper()]
mri_type = mri_types[1]

'''
Data Organization
'''

# Retrieving patient information
patient_data = retrieve_data(file_name)

total_patients = len(patient_data)

# sort_data(patient_data, sort_number)

img_names, label_data = crawl_for_images(mri_type, patient_data, sort_number)


	
if ML:
	(images, labels) = load_data(img_names, label_data, max_size)
	
	# Max (max of array)/ middle (middle of array)
	max = int(images.shape[0]-1)
	middle = int(images.shape[0]/2)

	
	# Create train and test images & labels
	train_images = images[0:middle]
	train_labels = labels[0:middle]
	test_images = images[int(middle+1):max]
	test_labels = labels[int(middle+1):max]
	
	# Begin Modelling
	model = setup_model()
	train_images = train_images / 255.0
	test_images = test_images / 255.0
	
	model.fit(train_images, train_labels, validation_data=(test_images,test_labels), epochs = total_epochs, batch_size = total_batch_size, verbose = 2)
	
	predictions = model.predict(test_images)
	normalized_predictions = []
	normalize_range = 30
	counter = 0
	correct_predictions = 0
	
	errors = []

	for prediction in predictions:
		max = np.amax(prediction)
		outcomes = np.where(prediction == max)[0][0]
		if not(isinstance(outcomes, list)):
			outcomes = [outcomes]
		for outcome in outcomes:
			actual = test_labels[counter]
			predicted = outcome
			
			error = (abs(predicted - actual))/actual
			
			errors.append(error)
			counter += 1
	total = counter
	
	print((np.average(errors)*100))
		
