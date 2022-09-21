# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:37:46 2022

@author: Dipto

This script seperates the classes of curated trainsets into their seperate classes
"""

from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

# loading in the ground truth labels
labels = pd.read_csv(r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\train.csv')


# reading in the datasets and the locations where we want the classes to be saved
# seperately
train_dir_60 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\60\images_with_60_dups_removed'
DFU_60 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\60\DFU_60"

train_dir_65 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\65\images_with_65_dups_removed'
DFU_65 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\65\DFU_65"

train_dir_70 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\70\images_with_70_dups_removed'
DFU_70 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\70\DFU_70"

train_dir_75 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\75\images_with_75_dups_removed'
DFU_75 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\75\DFU_75"

train_dir_80 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\80\images_with_80_dups_removed'
DFU_80 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\80\DFU_80"

train_dir_85 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\85\images_with_85_dups_removed'
DFU_85 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\85\DFU_85"

train_dir_90 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\90\images_with_90_dups_removed'
DFU_90 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\90\DFU_90"

train_dir_95 =r'C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\95\images_with_95_dups_removed'
DFU_95 = r"C:\Users\Dipto\OneDrive - MMU\Desktop\Train_Validation_Generator\95\DFU_95"


# custom function to seperate the classes
def class_seperator(file, store, img_dir):
    if not os.path.exists(store):
        os.mkdir(store)
        
    for image, none, infection, ischeamia, both in file.values:
        # Create subdirectory with classes
        if not os.path.exists(store + str(none) + str(infection) + str(ischeamia) + str(both)):
            os.mkdir(store + str(none) + str(infection) + str(ischeamia) + str(both))
        src_path = img_dir + '/' + image
        dst_path = store + str(none) + str(infection) + str(ischeamia) + str(both) + '/' + image 
        try:
            shutil.copy(src_path, dst_path)
            print("Copied Successfully!")
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When tried to copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))
            
        
# storing the results into different directories        
class_seperator(labels, DFU_60, train_dir_60)
class_seperator(labels, DFU_65, train_dir_65)
class_seperator(labels, DFU_70, train_dir_70)
class_seperator(labels, DFU_75, train_dir_75)
class_seperator(labels, DFU_80, train_dir_80)
class_seperator(labels, DFU_85, train_dir_85)
class_seperator(labels, DFU_90, train_dir_90)
class_seperator(labels, DFU_95, train_dir_95)



        

        