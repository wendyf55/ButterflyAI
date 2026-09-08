# for labelling the monarch images to create a training set

#I downloaded a bunch of monarch images, then manually sorted them into two separate folders for feeding and non-feeding
#Create a csv of each image filename and what their class is based on which folder they were in 

import os
import shutil
import pandas as pd
import numpy as np


os.chdir('/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images')

#test_df = pd.read_csv('Monarch_images.csv')
#this dataframe already has a column with the urls of each images, but not the class labels
test_df = pd.read_csv('Monarch_image_predictions.csv')

# Source folder containing unlabeled images
#source_folder = "/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images"

# Destination folders
#class0_folder = "Class_0"
#class1_folder = "Class_1"

#os.makedirs(class0_folder, exist_ok=True)
#os.makedirs(class1_folder, exist_ok=True)

# Move images based on prediction
# for _, row in test_df.iterrows():
#    filename = row['FileName']
#    prediction = row['prediction']

#    destination_path = os.path.join(source_folder, filename)

#    if prediction == 1:
#        source_path = os.path.join(class1_folder, filename)
#    else:
#        source_path = os.path.join(class0_folder, filename)

#    shutil.move(source_path, destination_path)

print("Images moved successfully (not)")

labels_df = pd.read_csv('Monarch_non_feeding_ls.csv')

test_df["Label"] = np.where(test_df["FileName"].isin(labels_df["FileName"]), "Non_feeding", "Feeding")

test_df.to_csv('/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images/Monarch_image_labels.csv', encoding='utf-8', index=False)

print("Done")