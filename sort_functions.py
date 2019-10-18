import csv
from PIL import Image, ImageFont, ImageDraw
from os import listdir
from os.path import isfile, join
import os, fnmatch

# Finds all file_types(i.e gif, mp4, mp3) in folder_name (sample_folder/sample_folder/) inside of child directories

def move_file_type(file_type, folder_name):
  try:
    os.mkdir(folder_name)
  except:
    print("A folder named: \"" + folder_name + "\" already exists.")
  for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
      if (name.find(file_type) != -1):
        curr_file = os.path.join(root,name)
        new_file = "./" + folder_name +"/" + name
          
        os.rename(curr_file,new_file)
		
# Finds all file_types(i.e gif, mp4, mp3) of file_name in folder_name

def move_file_name_type(file_name, file_type, folder_name):
  try:
    os.mkdir(folder_name)
  except:
    print("A folder named: \"" + folder_name + "\" already exists.")
  for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
      if ((name.find(file_type) != -1) and (name.find(file_name) != -1)):
        curr_file = os.path.join(root,name)
        new_file = "./" + folder_name +"/" + name
          
        os.rename(curr_file,new_file)

# retrives data from csv file named file_name

def retrieve_data(file_name):
	patient_data = []
	with open(file_name, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			MMSE = row['MMSE']
			CDR = row['CDR']
			ID = row['ID'].split("_")[1]
			if(CDR != '' or MMSE != ''):
				patient_data.append([row['ID'], [row['MMSE'], row['CDR']]])
	return patient_data

# Sorts data by MMSE, CDR or ID

def sort_data(patient_data, sort_number):
	total_patients = len(patient_data)
	
	for list_pos in range(total_patients):
		for curr_patient_pos in range(0, total_patients - list_pos - 1):
			curr_patient = patient_data[curr_patient_pos]
			next_patient = patient_data[curr_patient_pos + 1]
			curr_patient_data = float(curr_patient[1][sort_number])
			next_patient_data = float(next_patient[1][sort_number])
			if curr_patient_data > next_patient_data:
				patient_data[curr_patient_pos], patient_data[curr_patient_pos + 1] = next_patient, curr_patient

# Searches for pattern in path				
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.find(pattern) != -1:
                result.append(os.path.join(root, name))
    return result

# Searches for image data for patient passed in
def crawl_for_images(mri_type, patient_data, sort_number):
	image_paths = []
	label_data = []
	for patient in patient_data:
		id = patient[0]
		image_paths.append([find(id, mri_type)])
		label_data.append(patient[1][sort_number])
	return image_paths, label_data

# Creates a gif of all patients
def create_gif(image_paths,patient_data, sort_number,output_file_name):
	imgs = []
	counter = 0
	last_data = ""
	for image_path in image_paths:
		curr_data = image_path[1]
		label = "MMSE: " + str(curr_data)
		if curr_data != last_data:
			try:
				img = Image.open(image_path[0][0])
				draw = ImageDraw.Draw(img)
				font = ImageFont.truetype("arial.ttf", 15)
				draw.text((0, 0),label, fill='white',font=font)
				imgs.append(img)
			except:
				error = 1
		counter += 1
	imgs[0].save(output_file_name,
               save_all=True,
               append_images=imgs[1:],
               duration=100,
               loop=0)