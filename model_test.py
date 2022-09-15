#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:06:25 2022

@author: bc
"""

# Importing the rquired libraries
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Add, Dense, Dropout, Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
tf.random.set_seed(42)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())
#print(keras.__version__)
print(tf.__version__)
print(np.__version__)


# Source path of the validation and test images
src_path_test = "/home/bc/Documents/Imran/Imran_DFU_Project/DFUC2021_test/test"
src_path_val = "/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/val"


# loading and scalling the test images
test_datagen = ImageDataGenerator(rescale=1 / 255.0,fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    color_mode="rgb",
    target_size = (224,224),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)    
    
val_datagen = ImageDataGenerator(rescale=1 / 255.0, fill_mode='nearest') 
    
valid_generator = val_datagen.flow_from_directory(
    directory=src_path_val,
    color_mode="rgb",
    batch_size=32,
    target_size = (224,224),
    class_mode="categorical",
    shuffle=True,
    seed=42
) 



# loading the model and generating predicted probabilites of each of the classes
model_test = load_model("IncpBaseline.h5")

# evaluating the model on the validation data
model_test.evaluate(valid_generator, steps=valid_generator.n//valid_generator.batch_size)
test_generator.reset()

# making predictions on the test images
pred_class = model_test.predict(test_generator,steps=test_generator.n//test_generator.batch_size)
#cl = np.round(pred_class)

'''
predicted_class_indices = np.argmax(pred_class, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
'''

# saving the predictions into a dataframe according to class names
# I have used this order because in the train and val directories the classes are
# seperated into folders containing these names in this order
df = pd.DataFrame(pred_class, columns=['both', 'infection', 'ischeamia', 'none'])

# Get filenames of the test set
filenames=test_generator.filenames

#converting the filenames into dataframe and fixing the  names
df2 = pd.DataFrame(filenames, columns = ['image'])
df2 = df2["image"].str.replace("[all_classes/]","")

#df = df.assign(image=df2)
#df = df[['image','none','infection','ischeamia','both']]



# creating a dictionary of the dataframes to store them into the required format
d = {'image':df2, 
     'none':df['none'], 
     'infection':df['infection'],
     'ischeamia':df['ischeamia'],
     'both':df['both']}

# using that dictionary to create a dataframe of the results
result = pd.DataFrame(d)


# saving the results into a csv file
result.to_csv("Incp_baseline.csv",index=False)