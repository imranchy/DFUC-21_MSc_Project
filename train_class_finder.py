# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:35:31 2022

@author: Dipto

This script will identify the different classes of the DFUs in the Full Train set and also the similar images found
from the similarity tests carried proviously based on the CSV files created from the previous experiment and store the images
into different directories
"""

# Importing the required libraries
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

# Reading in the train.csv file which contains all the classes of the DFUs
train_df = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\train.csv')

#Creating a csv file of similar images with a similarity bracket of 70
train_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\70\train_threshold_70.csv')
train_70_images = pd.merge(train_df, train_70, left_on='image', right_on='image')
train_70_images = train_70_images.drop(columns=['Group ID','Folder', 'Size (KB)', 'Dimensions','Match %'])

train_70_images.to_csv('train_70_images.csv',index=False)

#Creating a csv file of similar images with a similarity bracket of 75
train_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\75\train_threshold_75.csv')
train_75_images = pd.merge(train_df, train_75, left_on='image', right_on='image')
train_75_images = train_75_images.drop(columns=['Group ID','Folder', 'Size (KB)', 'Dimensions','Match %'])

train_75_images.to_csv('train_75_images.csv',index=False)

#Creating a csv file of similar images with a similarity bracket of 80
train_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\80\train_threshold_80.csv')
train_80_images = pd.merge(train_df, train_80, left_on='image', right_on='image')
train_80_images = train_80_images.drop(columns=['Group ID','Folder', 'Size (KB)', 'Dimensions','Match %'])

train_80_images.to_csv('train_80_images.csv',index=False)

#Creating a csv file of similar images with a similarity bracket of 85
train_85 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\85\train_threshold_85.csv')
train_85_images = pd.merge(train_df, train_85, left_on='image', right_on='image')
train_85_images = train_85_images.drop(columns=['Group ID','Folder', 'Size (KB)','Dimensions','Match %'])

train_85_images.to_csv('train_85_images.csv',index=False)

#Creating a csv file of similar images with a similarity bracket of 90
train_90 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\90\train_threshold_90.csv')
train_90_images = pd.merge(train_df, train_90, left_on='image', right_on='image')
train_90_images = train_90_images.drop(columns=['Group ID','Folder', 'Size (KB)', 'Dimensions','Match %'])

train_90_images.to_csv('train_90_images.csv',index=False)

#Creating a csv file of similar images with a similarity bracket of 95
train_95 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\95\train_threshold_95.csv')
train_95_images = pd.merge(train_df, train_90, left_on='image', right_on='image')
train_95_images = train_95_images.drop(columns=['Group ID','Folder', 'Size (KB)', 'Dimensions','Match %'])

train_95_images.to_csv('train_95_images.csv',index=False)

#Reading in the created CSV files
full_set = train_df
sim_70 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_70_images.csv')
sim_75 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_75_images.csv')
sim_80 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_80_images.csv')
sim_85 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_85_images.csv')
sim_90 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_90_images.csv')
sim_95 = pd.read_csv(r'D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\train_95_images.csv')

# Loading in the directory of the train images
train_dir = r'D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\images'

# Creating directories to store the images
DFU_Full_Directory = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\DFU_Full_Train_Set"
DFU_70 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\70\DFU_70"
DFU_75 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\75\DFU_75"
DFU_80 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\80\DFU_80"
DFU_85 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\85\DFU_85"
DFU_90 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\90\DFU_90"
DFU_95 = r"D:\MSc Studies\MSc Project\DFU Internship\Curation Project\Similarity Results\train_similar_classes\95\DFU_95"

# A custome function to check from each of the csv files and store the images to the different class folder locations
def class_seperator(dfu, labels):
    if not os.path.exists(dfu):
        os.mkdir(dfu)
    for filename, none, infection, ischeamia, both in labels.values:
        # Create subdirectory with classes
        if not os.path.exists(dfu + str(none) + str(infection) + str(ischeamia) + str(both) ):
            os.mkdir(dfu + str(none) + str(infection) + str(ischeamia) + str(both) )
        src_path = train_dir + '/'+ filename
        dst_path = dfu + str(none) + str(infection) + str(ischeamia) + str(both) + '/' + filename
        try:
            shutil.copy(src_path, dst_path)
            print("Copied Successfully!")
        except IOError as e:
            print('Unable to copy file {} to {}'.format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
    
# Storing the images into seperate class folders 
class_seperator(DFU_Full_Directory, full_set)
class_seperator(DFU_70, sim_70)
class_seperator(DFU_75, sim_75)
class_seperator(DFU_80, sim_80)
class_seperator(DFU_85, sim_85)
class_seperator(DFU_90, sim_90)
class_seperator(DFU_95, sim_95)

















                       