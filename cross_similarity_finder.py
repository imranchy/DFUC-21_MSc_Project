# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:50:04 2022

@author: Dipto

This script enables to idenfity cross similar images that exists between the classes from the similarity
thresholds used to run the Image Hashing Algorithm run by using dupeGuru
"""

#Importing pandas to read, modify and create new CSV files based
import pandas as pd

# Creating modified CSV files for each of the Thresholds
# for both vs none classes to identify images that might exist between these classes
# and saving the results into csv files so that we can find the images from the filenames
both_and_none_th_70 = pd.read_csv("both_and_none_70.csv")
both_and_none_th_70 = both_and_none_th_70[both_and_none_th_70.groupby(['Group ID'])['Folder'].transform('nunique')>1]
both_and_none_th_70.to_csv('both_and_none_th_70_modified.csv',index=False)

both_and_none_th_75 = pd.read_csv("both_and_none_75.csv")
both_and_none_th_75 = both_and_none_th_75[both_and_none_th_75.groupby(['Group ID'])['Folder'].transform('nunique')>1]
both_and_none_th_75.to_csv('both_and_none_th_75_modified.csv',index=False)

both_and_none_th_80 = pd.read_csv("both_and_none_80.csv")
both_and_none_th_80 = both_and_none_th_80[both_and_none_th_80.groupby(['Group ID'])['Folder'].transform('nunique')>1]
both_and_none_th_80.to_csv('both_and_none_th_80_modified.csv',index=False)

both_and_none_th_85 = pd.read_csv("both_and_none_85.csv")
both_and_none_th_85 = both_and_none_th_85[both_and_none_th_85.groupby(['Group ID'])['Folder'].transform('nunique')>1]
both_and_none_th_85.to_csv('both_and_none_th_85_modified.csv',index=False)


# Creating modified CSV files for each of the Thresholds
# for infection vs ischaemia classes to identify images that might exist between these classes
# and saving the results into csv files so that we can find the images from the filenames
infection_ischaemia_th_70 = pd.read_csv("infection_ischaemia_70.csv")
infection_ischaemia_th_70 = infection_ischaemia_th_70[infection_ischaemia_th_70.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_ischaemia_th_70.to_csv('infection_ischaemia_th_70_modified.csv',index=False)

infection_ischaemia_th_75 = pd.read_csv("infection_ischaemia_75.csv")
infection_ischaemia_th_75 = infection_ischaemia_th_75[infection_ischaemia_th_75.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_ischaemia_th_75.to_csv('infection_ischaemia_th_75_modified.csv',index=False)

infection_ischaemia_th_80 = pd.read_csv("infection_ischaemia_80.csv")
infection_ischaemia_th_80 = infection_ischaemia_th_80[infection_ischaemia_th_80.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_ischaemia_th_80.to_csv('infection_ischaemia_th_80_modified.csv',index=False)

infection_ischaemia_th_85 = pd.read_csv("infection_ischaemia_85.csv")
infection_ischaemia_th_85 = infection_ischaemia_th_85[infection_ischaemia_th_85.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_ischaemia_th_85.to_csv('infection_ischaemia_th_85_modified.csv',index=False)

infection_ischaemia_th_90 = pd.read_csv("infection_ischaemia_90.csv")
infection_ischaemia_th_90 = infection_ischaemia_th_90[infection_ischaemia_th_90.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_ischaemia_th_90.to_csv('infection_ischaemia_th_90_modified.csv',index=False)



# Creating modified CSV files for each of the Thresholds
# for infection vs none classes to identify images that might exist between these classes
# and saving the results into csv files so that we can find the images from the filenames
infection_none_th_70 = pd.read_csv("infection_none_70.csv")
infection_none_th_70 = infection_none_th_70[infection_none_th_70.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_none_th_70.to_csv('infection_none_th_70_modified.csv',index=False)

infection_none_th_75 = pd.read_csv("infection_none_75.csv")
infection_none_th_75 = infection_none_th_75[infection_none_th_75.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_none_th_75.to_csv('infection_none_th_75_modified.csv',index=False)

infection_none_th_80 = pd.read_csv("infection_none_80.csv")
infection_none_th_80 = infection_none_th_80[infection_none_th_80.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_none_th_80.to_csv('infection_none_th_80_modified.csv',index=False)

infection_none_th_85 = pd.read_csv("infection_none_85.csv")
infection_none_th_85 = infection_none_th_85[infection_none_th_85.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_none_th_85.to_csv('infection_none_th_85_modified.csv',index=False)

infection_none_th_90 = pd.read_csv("infection_none_90.csv")
infection_none_th_90 = infection_none_th_90[infection_none_th_90.groupby(['Group ID'])['Folder'].transform('nunique')>1]
infection_none_th_90.to_csv('infection_none_th_90_modified.csv',index=False)



# Creating modified CSV files for each of the Thresholds
# for ischaemia vs none classes to identify images that might exist between these classes
# and saving the results into csv files so that we can find the images from the filenames
ischaemia_none_th_70 = pd.read_csv("ischaemia_none_70.csv")
ischaemia_none_th_70 = ischaemia_none_th_70[ischaemia_none_th_70.groupby(['Group ID'])['Folder'].transform('nunique')>1]
ischaemia_none_th_70.to_csv('ischaemia_none_th_70_modified.csv',index=False)

ischaemia_none_th_75 = pd.read_csv("ischaemia_none_75.csv")
ischaemia_none_th_75 = ischaemia_none_th_75[ischaemia_none_th_75.groupby(['Group ID'])['Folder'].transform('nunique')>1]
ischaemia_none_th_75.to_csv('ischaemia_none_th_75_modified.csv',index=False)

ischaemia_none_th_80 = pd.read_csv("ischaemia_none_80.csv")
ischaemia_none_th_80 = ischaemia_none_th_80[ischaemia_none_th_80.groupby(['Group ID'])['Folder'].transform('nunique')>1]
ischaemia_none_th_80.to_csv('ischaemia_none_th_80_modified.csv',index=False)

ischaemia_none_th_85 = pd.read_csv("ischaemia_none_85.csv")
ischaemia_none_th_85 = ischaemia_none_th_85[ischaemia_none_th_85.groupby(['Group ID'])['Folder'].transform('nunique')>1]
ischaemia_none_th_85.to_csv('ischaemia_none_th_85_modified.csv',index=False)









