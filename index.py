'''
[['patient no. 1', [mmse, cdr]],['patient no. 2', [mmse, cdr]], etc. ]
MMSE = Mini-Mental State Exam
CDR = Clinical Dementia Rating
'''
		
from sort_functions import retrieve_data, sort_data, crawl_for_images
from image_recognition import load_data, setup_model, normalize

import tensorflow as tf
from tensorflow import keras
				
file_name = "oasis_cross_sectional.csv"	
sort_name = "MMSE" # Either CDR or MMSE
mri_types = ["t88_gfc_tra", "t88_masked_gfc_tra", "t88_masked_gfc_fseg_tra","t88_gfc_cor","111_t88_gfc_sag","111_sag_88","mpr-1_anon_sag_66","mpr-2_anon_sag_66","mpr-3_anon_sag_66","mpr-4_anon_sag_66"]

max_size = 100, 100

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
	(train_images, train_labels) = load_data(img_names, label_data, max_size)
	print(train_images.shape)
	print(train_labels.shape)
	model = setup_model()
	train_images = train_images / 255.0
	test_images = train_images
	test_labels = train_labels
	model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs = 600, batch_size = 31, verbose = 2)
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("Baseline Error: %.2f%%" % (100-scores[1]*100))
	
	predictions = model.predict(test_images)
	normalized_predictions = []
	normalize_range = 30
	for prediction in predictions:
		normalized_predictions.append(normalize(prediction, normalize_range))
	print(normalized_predictions)
	
		
