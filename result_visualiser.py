# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:23:43 2022

@author: Dipto
"""

import pandas as pd
import numpy as np
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
#fig.suptitle('Number of the Classes of the DFUC 2021 Dataset', fontsize=16)

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

# Plotting the validation accuracy scores of the best model on each datasets
X = ['InceptionResNetV2 (full)', 'InceptionResNetV2 (60)','InceptionResNetV2 (65)',
     'Xception (70)', 'InceptionResNetV2 (75)', 'InceptionResNetV2 (80)',
     'InceptionResNetV2 (85)','InceptionResNetV2 (90)', 'ResNet152 (95)']
y = [80,47,82,79,86,89,81,79,82]

# setting bar colours
colors = ['#A5C1DC' if (s < max(y)) else '#007CC3' for s in y]

# plotting the bargraph
plt.figure(figsize=(20,8))
plt.barh(X, y, color=colors)
plt.xlabel('Validation Accuracy')
plt.ylabel('Top Models on Each Trainsets')
plt.show()

def acc_plotter(x1, x2, y1, y2, lbl1, lbl2):
    '''
        Helper function to plot the Validation Accuracies
    '''
    plt.figure(figsize=(8,6))
    a = plt.subplot(1,1,1)
    a.plot(x1,y1,label=lbl1)
    a.legend(loc='lower right')
    a.plot(x2,y2,label=lbl2)
    a.legend(loc='lower right')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    
def loss_plotter(x1, x2, y1, y2, lbl1, lbl2):
    '''
        Helper Function to plot the validation losses
    '''
    plt.figure(figsize=(8,6))
    a = plt.subplot(1,1,1)
    a.plot(x1,y1,label=lbl1)
    a.legend(loc='upper right')
    a.plot(x2,y2,label=lbl2)
    a.legend(loc='upper right')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Loss")
    
# This part is for the full dataset
model1 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/IncpFull.csv')
model6 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp80.csv')

# number of epochs for the top models
x1 = model1['epoch']
x2 = model6['epoch']

# val accuracies
y1 = model1['val_accuracy']
y2 = model6['val_accuracy']

# plotting the graph
acc_plotter(x1, x2, y1, y2,'InceptionResNetV2_full', 'InceptionResNetV2_80')

# validation losses
y1 = model1['val_loss']
y2 = model6['val_loss']

# plotting the graph
loss_plotter(x1, x2, y1, y2, 'InceptionResNetV2_full', 'InceptionResNetV2_80')

# Plotting the macro scores
X = ['F1-Score','Precision','Recall','AUC']
Y = [51,52,54,84]
Z = [53,55,55,84]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y, 0.4, label = 'InceptionResNetV2 (full)', color='#A5C1DC')
plt.bar(X_axis+0.2,Z,0.4, label = 'InceptionResNetV2 (80)', color='#007CC3')
  
plt.xticks(X_axis, X)
plt.legend()
plt.show()


# Plotting the class perfromances
X = ['None','Infection','Ischeamia','Both','Accuracy']
Y = [70,51,43,39,60]
Z = [72,54,45,43,62]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y, 0.4, label = 'InceptionResNetV2 (full)', color='#A5C1DC')
plt.bar(X_axis+0.2,Z,0.4, label = 'InceptionResNetV2 (80)', color='#007CC3')

plt.xticks(X_axis, X)
plt.legend()
plt.show()



































