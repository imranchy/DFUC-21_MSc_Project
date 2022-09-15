# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:00:26 2022

This script will check the image group id of each of the similarity brackets then merge them with the train.csv
file and pull the images from the training images set and place them into different folders

@author: Dipto
"""

# Importing the required libraries
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

# Reading in the train.csv file which contains all the classes of the DFUs
train_df = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\train.csv')

# Loading in the directory of the train images
train_dir = r'D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\images'


#Reading CSV file containing the duplicate images, merging it to the ground truth values,
# then dropping the duplicate images based on the Group ID
# and storing the results into seperate csv files for each of the 4 classes

# These are for the Both 70
both_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images/both_70.csv')
both_70_images = pd.merge(train_df, both_70, left_on='image', right_on='image')
both_70_images = both_70_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

both_70_images = both_70_images.drop_duplicates(subset=['Group ID'], keep='last')
both_70_images = both_70_images.drop(columns=['Group ID'])
both_70_images.to_csv('both_70_modified.csv',index=False)

#Reading in the created CSV files
both_dup_70 = pd.read_csv('both_70_modified.csv')

# Reading in directories to store the images
both_70 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images\both_70_images"


# Both 75
both_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images/both_75.csv')
both_75_images = pd.merge(train_df, both_75, left_on='image', right_on='image')
both_75_images = both_75_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

both_75_images = both_75_images.drop_duplicates(subset=['Group ID'], keep='last')
both_75_images = both_75_images.drop(columns=['Group ID'])
both_75_images.to_csv('none_75_modified.csv',index=False)

#Reading in the created CSV files
both_dup_75 = pd.read_csv('none_75_modified.csv')

# Reading in directories to store the images
both_75 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images\both_75_images"


# Both 80
both_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images/both_80.csv')
both_80_images = pd.merge(train_df, both_80, left_on='image', right_on='image')
both_80_images = both_80_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

both_80_images = both_80_images.drop_duplicates(subset=['Group ID'], keep='last')
both_80_images = both_80_images.drop(columns=['Group ID'])
both_80_images.to_csv('both_80_modified.csv',index=False)

#Reading in the created CSV files
both_dup_80 = pd.read_csv('both_80_modified.csv')

# Reading in directories to store the images
both_80 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\both_images\both_80_images"


# None 70
none_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images/none_70.csv')
none_70_images = pd.merge(train_df, none_70, left_on='image', right_on='image')
none_70_images = none_70_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

none_70_images = none_70_images.drop_duplicates(subset=['Group ID'], keep='last')
none_70_images = none_70_images.drop(columns=['Group ID'])
none_70_images.to_csv('none_70_modified.csv',index=False)

#Reading in the created CSV files
none_dup_70 = pd.read_csv('none_70_modified.csv')

# Reading in directories to store the images
none_70 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images\none_70_images"


# None 75
none_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images/none_75.csv')
none_75_images = pd.merge(train_df, none_75, left_on='image', right_on='image')
none_75_images = none_75_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

none_75_images = none_75_images.drop_duplicates(subset=['Group ID'], keep='last')
none_75_images = none_75_images.drop(columns=['Group ID'])
none_75_images.to_csv('none_75_modified.csv',index=False)

#Reading in the created CSV files
none_dup_75 = pd.read_csv('none_75_modified.csv')

# Reading in directories to store the images
none_75 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images\none_75_images"


# None 80
none_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images/none_80.csv')
none_80_images = pd.merge(train_df, none_80, left_on='image', right_on='image')
none_80_images = none_80_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

none_80_images = none_80_images.drop_duplicates(subset=['Group ID'], keep='last')
none_80_images = none_80_images.drop(columns=['Group ID'])
none_80_images.to_csv('none_80_modified.csv',index=False)

#Reading in the created CSV files
none_dup_80 = pd.read_csv('none_80_modified.csv')

# Reading in directories to store the images
none_80 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\none_images\none_80_images"


# infection 70
infection_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images/infection_70.csv')
infection_70_images = pd.merge(train_df, infection_70, left_on='image', right_on='image')
infection_70_images = infection_70_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

infection_70_images = infection_70_images.drop_duplicates(subset=['Group ID'], keep='last')
infection_70_images = infection_70_images.drop(columns=['Group ID'])
infection_70_images.to_csv('infection_70_modified.csv',index=False)

#Reading in the created CSV files
infection_dup_70 = pd.read_csv('infection_70_modified.csv')

# Reading in directories to store the images
infection_70 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images\infection_70_images"


# infection 75
infection_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images/infection_75.csv')
infection_75_images = pd.merge(train_df, infection_75, left_on='image', right_on='image')
infection_75_images = infection_75_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

infection_75_images = infection_75_images.drop_duplicates(subset=['Group ID'], keep='last')
infection_75_images = infection_75_images.drop(columns=['Group ID'])
infection_75_images.to_csv('infection_75_modified.csv',index=False)

#Reading in the created CSV files
infection_dup_75 = pd.read_csv('infection_75_modified.csv')

# Reading in directories to store the images
infection_75 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images\infection_75_images"


# infection 80
infection_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images/infection_80.csv')
infection_80_images = pd.merge(train_df, infection_80, left_on='image', right_on='image')
infection_80_images = infection_80_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

infection_80_images = infection_80_images.drop_duplicates(subset=['Group ID'], keep='last')
infection_80_images = infection_80_images.drop(columns=['Group ID'])
infection_80_images.to_csv('infection_80_modified.csv',index=False)

#Reading in the created CSV files
infection_dup_80 = pd.read_csv('infection_80_modified.csv')

# Reading in directories to store the images
infection_80 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\infection_images\infection_80_images"


# ischaemia 70
ischaemia_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images/ischaemia_70.csv')
ischaemia_70_images = pd.merge(train_df, ischaemia_70, left_on='image', right_on='image')
ischaemia_70_images = ischaemia_70_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

ischaemia_70_images = ischaemia_70_images.drop_duplicates(subset=['Group ID'], keep='last')
ischaemia_70_images = ischaemia_70_images.drop(columns=['Group ID'])
ischaemia_70_images.to_csv('ischaemia_70_modified.csv',index=False)

#Reading in the created CSV files
ischaemia_dup_70 = pd.read_csv('ischaemia_70_modified.csv')

# Reading in directories to store the images
ischaemia_70 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images\ischaemia_70_images"


# ischaemia 75
ischaemia_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images/ischaemia_75.csv')
ischaemia_75_images = pd.merge(train_df, ischaemia_75, left_on='image', right_on='image')
ischaemia_75_images = ischaemia_75_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

ischaemia_75_images = ischaemia_75_images.drop_duplicates(subset=['Group ID'], keep='last')
ischaemia_75_images = ischaemia_75_images.drop(columns=['Group ID'])
ischaemia_75_images.to_csv('ischaemia_75_modified.csv',index=False)

#Reading in the created CSV files
ischaemia_dup_75 = pd.read_csv('ischaemia_75_modified.csv')

# Reading in directories to store the images
ischaemia_75 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images\ischaemia_75_images"


# ischaemia 80
ischaemia_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images/ischaemia_80.csv')
ischaemia_80_images = pd.merge(train_df, ischaemia_80, left_on='image', right_on='image')
ischaemia_80_images = ischaemia_80_images.drop(columns=['Folder', 'Size (KB)', 'Dimensions','Match %'])

ischaemia_80_images = ischaemia_80_images.drop_duplicates(subset=['Group ID'], keep='last')
ischaemia_80_images = ischaemia_80_images.drop(columns=['Group ID'])
ischaemia_80_images.to_csv('ischaemia_80_modified.csv',index=False)

#Reading in the created CSV files
ischaemia_dup_80 = pd.read_csv('ischaemia_80_modified.csv')

# Reading in directories to store the images
ischaemia_80 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Duplicates Removed\ischaemia_images\ischaemia_80_images"


# A custome function to check from each of the csv files and store the images to the different class folder locations
def class_seperator(dfu, labels):
    """
        The first parameter is the directory in which we are storing the images and the 2nd parameter is the 
        created CSV files which we are reading
    """
    if not os.path.exists(dfu):
        os.mkdir(dfu)
    for image, none, infection, ischeamia, both in labels.values:
        # Create subdirectory with classes
        if not os.path.exists(dfu + str(none) + str(infection) + str(ischeamia) + str(both) ):
            os.mkdir(dfu + str(none) + str(infection) + str(ischeamia) + str(both) )
        src_path = train_dir + '/'+ image
        dst_path = dfu + str(none) + str(infection) + str(ischeamia) + str(both) + '/' + image
        try:
            shutil.copy(src_path, dst_path)
            print("Copied Successfully!")
        except IOError as e:
            print('Unable to copy file {} to {}'.format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
            
            
# Storing the images into seperate folders 
class_seperator(both_70, both_dup_70)
class_seperator(both_75, both_dup_75)
class_seperator(both_80, both_dup_80)

class_seperator(infection_70, infection_dup_70)
class_seperator(infection_75, infection_dup_75)
class_seperator(infection_80, infection_dup_80)

class_seperator(ischaemia_70, ischaemia_dup_70)
class_seperator(ischaemia_75, ischaemia_dup_75)
class_seperator(ischaemia_80, ischaemia_dup_80)

class_seperator(none_70, none_dup_70)
class_seperator(none_75, none_dup_75)
class_seperator(none_80, none_dup_80)