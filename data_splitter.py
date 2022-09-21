# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:20:48 2022

@author: Dipto


This script splits the images stored in different folders based on the similarity brackets
into 80% training and 20% validation sets
"""


import splitfolders


set_60 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\60\\DFU_60"
out_60 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\60\DFU_60"

set_65 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\65\\DFU_65"
out_65 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\65\DFU_65"

set_70 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\70\\DFU_70"
out_70 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\70\DFU_70"

set_75 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\75\\DFU_75"
out_75 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\75\DFU_75"

set_80 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\80\\DFU_80"
out_80 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\80\DFU_80"

set_85 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\85\\DFU_85"
out_85 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\85\DFU_85"

set_90 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\90\\DFU_90"
out_90 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\90\DFU_90"

set_95 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\Train_Validation_Generator\\95\\DFU_95"
out_95 = "C:\\Users\\Dipto\OneDrive - MMU\\Desktop\\Train_Validation_Generator\\95\DFU_95"
         
     
splitfolders.ratio(input=set_60, output=out_60, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_65, output=out_65, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_70, output=out_70, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_75, output=out_75, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_80, output=out_80, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_85, output=out_85, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_90, output=out_90, seed=42, ratio=(0.8,0.2))
splitfolders.ratio(input=set_95, output=out_95, seed=42, ratio=(0.8,0.2))












