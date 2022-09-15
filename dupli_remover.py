# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 14:21:57 2022

@author: Dipto

This script file will identify the similar images of each threshold which exists between both train and test
sets of the Dataset. Then the results will be stored into seperate CSV files
"""

#Importing pandas to read, modify and create new CSV files based
import pandas as pd

# Creating modified CSV files for each of the Thresholds (70 to 95) and storing them in a CSV file according to their Group IDs
th_70 = pd.read_csv("train_test_threshold_70.csv")
th_70 = th_70[th_70.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_70.to_csv('train_test_threshold_70_modified.csv',index=False)

th_75 = pd.read_csv("train_test_threshold_75.csv")
th_75 = th_75[th_75.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_75.to_csv('train_test_threshold_75_modified.csv',index=False)  

th_80 = pd.read_csv("train_test_threshold_80.csv")
th_80 = th_80[th_80.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_80.to_csv('train_test_threshold_80_modified.csv',index=False)  

th_85 = pd.read_csv("train_test_threshold_85.csv")
th_85 = th_85[th_85.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_85.to_csv('train_test_threshold_85_modified.csv',index=False)  

th_90 = pd.read_csv("train_test_threshold_90.csv")
th_90 = th_90[th_90.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_90.to_csv('train_test_threshold_90_modified.csv',index=False)  

th_95 = pd.read_csv("train_test_threshold_95.csv")
th_95 = th_95[th_95.groupby(['Group ID'])['Folder'].transform('nunique')>1]
th_95.to_csv('train_test_threshold_95_modified.csv',index=False)  
  