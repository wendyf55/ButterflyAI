# for labelling the monarch images to create a training set

#I downloaded a bunch of monarch images, then manually sorted them into two separate folders for feeding and non-feeding
#Create a csv of each image filename and what their class is based on which folder they were in 

import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONARCH_DIR = PROJECT_ROOT / "data" / "Monarch_images"
PREDICTIONS_CSV = MONARCH_DIR / "Monarch_image_predictions.csv"
NON_FEEDING_CSV = MONARCH_DIR / "Monarch_non_feeding_ls.csv"
LABELS_CSV = MONARCH_DIR / "Monarch_image_labels.csv"
NON_FEEDING_LABEL = "Non_feeding"
FEEDING_LABEL = "Feeding"

#test_df = pd.read_csv('Monarch_images.csv')
#this dataframe already has a column with the urls of each images, but not the class labels
test_df = pd.read_csv(PREDICTIONS_CSV)

# Deprecated: this older workflow moved predicted images between class folders.
# Keeping the block for reference while labels are now assigned from the manual CSV.
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
labels_df = pd.read_csv(NON_FEEDING_CSV)

test_df["Label"] = np.where(test_df["FileName"].isin(labels_df["FileName"]), NON_FEEDING_LABEL, FEEDING_LABEL)

test_df.to_csv(LABELS_CSV, encoding='utf-8', index=False)

print(f"Saved labels to {LABELS_CSV}")
