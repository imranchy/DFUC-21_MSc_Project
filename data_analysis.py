# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:22:41 2022

This script shows some statistical analysis of the DFUC 2021 dataset with counts of the classes,
their distribution and sample of some of the train images belonging to each of the classes

@author: Dipto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import os 
import glob


def img_shower(folder, lst):
    """
        This function prints sample of images of each of the classes
        in the train folder and sample of images in test set
    """
    plt.figure(figsize=(12, 11))
    for i in range(12):
        plt.subplot(4, 4, i + 1)
        img = plt.imread(os.path.join(folder, lst[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()


# Reading the csv file with ground truth labels
train_df = pd.read_csv('train.csv')
print("Total Images in the Train Set:",len(train_df))

# Image analysis and plotting
train = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train"

# Creating a dataframe to stored with the number of images belonging to each classes 
# storing the images as lists
both_train = glob.glob(train+"/DFU_Both/*.jpg")
none_train = glob.glob(train+"/DFU_None/*.jpg")
infection_train = glob.glob(train+"/DFU_Infection/*.jpg")
ischaemia_train = glob.glob(train+"/DFU_Ischeamia/*.jpg")

# storing the len of lists into a dataframe
data = pd.DataFrame(np.concatenate([[0]*len(both_train),[1]*len(none_train),[2]*len(infection_train),[3]*len(ischaemia_train)]),columns=["class"])

# storing the number of image classes as integers
s1 = (data['class']==0).sum()
s2 = (data['class']==1).sum()
s3 = (data['class']==2).sum()
s4 = (data['class']==3).sum()

# using those integers to get the number of images of each class and the total number of unlabelled images
print("Total Images of Both Class: ",s1)
print("Total Images of None Class: ",s2)
print("Total Images of Infection Class: ",s3)
print("Total Images of Ischaemia Class: ",s4)

# total unlabelled images
print("Total Unlabelled Images in the Dataset: ",len(train_df)-(s1+s2+s3+s4))

# checking the test directory
test_dir = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_test"
test = os.listdir("D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_test")

# getting the total image count in the test folder
test_count = glob.glob(test_dir+"/*.jpg")
print("Total Images in the blind Test Set: ",len(test_count))

# total images in the dataset
print("Total Images in the DFUC 2021 Dataset: ",len(test_count)+len(train_df))

# Creating a number of countplots of the classes in the train set
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
fig.suptitle('Number of the Classes of the DFUC 2021 Dataset', fontsize=16)

columns = ['none', 'infection', 'ischaemia', 'both']
for i, col in enumerate(columns):
    graph = sns.countplot(x=train_df[col], ax=ax[i], palette="Blues_r")
    graph.bar_label(graph.containers[0])
    
# creating the pie chart to show class distribution
labels = 'Both', 'None', 'Infection', 'Ischaemia'
sizes = [s1,s2,s3,s4]
explode = (0, 0, 0.1, 0)

colors = ['#E6F69D','#A5C1DC','#007CC3','#AADEA7'] #7982B9  #007CC3

fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
    
# Creading seperate directories of the classes which will help us generate the sample of images of these classes
both = os.listdir("D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Both")
both_dir = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Both"

none = os.listdir("D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_None")
none_dir = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_None"

infection = os.listdir("D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Infection")
infection_dir = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Infection"

ischaemia = os.listdir("D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Ischeamia")
ischaemia_dir = "D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train/DFU_Ischeamia"

# sample of Both images    
img_shower(both_dir, both)

# sample of none images
img_shower(none_dir, none)

# sample of none images
img_shower(infection_dir, infection)

# sample of none images
img_shower(ischaemia_dir, ischaemia)

# sample of images in the test set
img_shower(test_dir, test)






    
