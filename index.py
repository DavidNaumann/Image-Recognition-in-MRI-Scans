'''
[['patient no. 1', [mmse, cdr]],['patient no. 2', [mmse, cdr]], etc. ]
MMSE = Mini-Mental State Exam
CDR = Clinical Dementia Rating
'''
		
from sort_functions import retrieve_data, sort_data, crawl_for_images
				
file_name = "oasis_cross_sectional.csv"	
sort_name = "MMSE" # Either CDR or MMSE

sort_dict = {
		"MMSE": 0,
		"CDR": 1
}

sort_number = sort_dict[sort_name.upper()]

'''
Data Organization
'''

# Retrieving patient information
patient_data = retrieve_data(file_name)

total_patients = len(patient_data)

print("Total patients: " + str(total_patients))

print(patient_data)

# sort_data(patient_data, sort_number)

image_data = crawl_for_images(sort_name, patient_data, sort_number)
print(image_data)

		