# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 11:53:43 2022
In this script file I am checking all the obtained csv files under the similarity brackets
containing each of the class of images and comparing them against the original csv file (train.csv)
which contains the ground truth labels of each of the classes to perfrom cross checking and finally storing
the results into seperate csv files.

@author: Dipto
"""

import pandas as pd

# I am loading the train.csv file (the ground truth)
train_df = pd.read_csv("train.csv")

# here I am loading each of the csv files containg images belonging to each similar brackets
train_70 = pd.read_csv("train_70_images.csv")
train_75 = pd.read_csv("train_75_images.csv")
train_80 = pd.read_csv("train_80_images.csv")
train_85 = pd.read_csv("train_85_images.csv")
train_90 = pd.read_csv("train_90_images.csv")
train_95 = pd.read_csv("train_95_images.csv")

# checking and storing the results into seperate csv files
result_70 = train_df[train_df.apply(tuple,1).isin(train_70.apply(tuple,1))]
result_70.to_csv("result_70.csv")

result_75 = train_df[train_df.apply(tuple,1).isin(train_75.apply(tuple,1))]
result_75.to_csv("result_75.csv")

result_80 = train_df[train_df.apply(tuple,1).isin(train_80.apply(tuple,1))]
result_80.to_csv("result_80.csv")

result_85 = train_df[train_df.apply(tuple,1).isin(train_85.apply(tuple,1))]
result_85.to_csv("result_85.csv")

result_90 = train_df[train_df.apply(tuple,1).isin(train_90.apply(tuple,1))]
result_90.to_csv("result_90.csv")

result_95 = train_df[train_df.apply(tuple,1).isin(train_95.apply(tuple,1))]
result_95.to_csv("result_95.csv")