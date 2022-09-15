# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:01:01 2022

@author: Dipto
"""

from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

labels = pd.read_csv(r'train.csv')


train_dir =r'D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\images'
DFU = r"D:\MSc Studies\MSc Project\MSc Work\MS_Project\Imran_DFU_2021_Dataset\Imran\DFUC2021_train\DFU"
if not os.path.exists(DFU):
    os.mkdir(DFU)

for filename, none, infection, ischeamia, both in labels.values:
    # Create subdirectory with classes
    if not os.path.exists(DFU + str(none) + str(infection) + str(ischeamia) + str(both) ):
        os.mkdir(DFU + str(none) + str(infection) + str(ischeamia) + str(both) )
    src_path = train_dir + '/'+ filename
    dst_path = DFU + str(none) + str(infection) + str(ischeamia) + str(both) + '/' + filename
    try:
        shutil.copy(src_path, dst_path)
        print("Copied Successfully!")
    except IOError as e:
        print('Unable to copy file {} to {}'.format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))