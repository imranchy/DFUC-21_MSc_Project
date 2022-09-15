#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:21:45 2022

@author: bc
"""

#import keras 
import tensorflow as tf  
from tensorflow import keras
#import cv2 as cv  
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time 
#import pickle
import re
import tensorflow_addons 

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Add, Dense, Dropout, Layer, Input, Flatten, LayerNormalization, MultiHeadAttention, Embedding
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import AdamW
#from tensorflow.keras.utils.generic_utils import Progbar  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers

#from keras.utils import multi_gpu_model
#import pandas as pd
tf.random.set_seed(42)

print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())
print(tf.__version__)
print(np.__version__)

# array of models we are  training
models_to_train = ['InceptionResNetV2']
batch_sizes = [32]

optimisers = ['AdamW']

class_count = 4
training_loops = 1
#input_shape = (32,32,3)
#max_size = 72
save_path = r"/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/dfu_80_vit_results"
num_gpus = 1
num_epochs = 200
# set this to imagenet to pretrain on that
weights = None#"imagenet"
learning_rate = 0.001
weight_decay = 0.0001


image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

src_path_train = "/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/train"
src_path_val = "/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/val"

# scaling and augmentation
def getDataGens(batch_size):
    train_datagen = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest")
        
    val_datagen = ImageDataGenerator(
            rescale=1 / 255.0,
            fill_mode="nearest") 
    
    train_generator = train_datagen.flow_from_directory(
        directory=src_path_train,
        color_mode="rgb",
        batch_size=batch_size,
        target_size = (image_size,image_size),
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
    )
    
    valid_generator = val_datagen.flow_from_directory(
        directory=src_path_val,
        color_mode="rgb",
        batch_size=batch_size,
        target_size = (image_size, image_size),
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
    ) 
    return train_datagen, val_datagen, train_generator, valid_generator

# The MLP
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

# This is the Patches class
class Patches(Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({'patch_size': self.patch_size})
        return config
    
# This is the Patchencoder class
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(units=self.projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )
        
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            'num_patches':self.num_patches,
            'projection_dim':self.projection_dim
            #'projection':self.projection,
            #'position_embedding':self.position_embedding
            })
        return config
                     
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    
for model_count in range(len(models_to_train)):
    model_type = models_to_train[model_count] 
    if not os.path.exists(os.path.join(save_path,model_type)):
            os.mkdir(os.path.join(save_path,model_type))
    currSavePath = os.path.join(save_path,model_type)
    for opt_count in range(len(optimisers)):
        opt_type = optimisers[opt_count]
        if not os.path.exists(os.path.join(currSavePath,opt_type)):
            os.mkdir(os.path.join(currSavePath,opt_type))
        currSavePath = os.path.join(currSavePath,opt_type)
        model = None 
        for batchSizes in range (len(batch_sizes)): 
            tf.keras.backend.clear_session()
            if not os.path.exists(os.path.join(currSavePath,str(batch_sizes[batchSizes]))):
                os.mkdir(os.path.join(currSavePath,str(batch_sizes[batchSizes])))
            currSavePath = os.path.join(currSavePath,str(batch_sizes[batchSizes]))
            train_datagen, val_datagen, train_generator, valid_generator = getDataGens(batch_sizes[batchSizes])
            model_func = getattr(tf.keras.applications,model_type) 
            model = model_func(include_top=False,
                weights=None,
                input_shape=None,
                input_tensor=None,
                pooling=None
                )
            
            inputs = model.input
            #augmented = data_augmentation(intake)
            patches = Patches(patch_size)(inputs)
            encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
            
            
            for _ in range(transformer_layers):
                x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
                attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
                )(x1, x1)
                x2 = Add()([attention_output, encoded_patches])
                x3 = LayerNormalization(epsilon=1e-6)(x2)
                x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
                encoded_patches = Add()([x3, x2])
                
            representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = Flatten()(representation)
            representation = Dropout(0.5)(representation)
            features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
            logits = Dense(class_count)(features)
            
            model = Model(inputs=inputs, outputs=logits)

            model.summary()
            tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
            config = model.get_config()
            # At loading time, register the custom objects with a `custom_object_scope`:
            custom_objects = {"Patches":Patches, "PatchEncoder":PatchEncoder}
            with tf.keras.utils.custom_object_scope(custom_objects):
                new_model = tf.keras.Model.from_config(config)


            csv_logger =  [
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(currSavePath,model_type+'_batchSize_'+str(batchSizes)+"_opt_"+opt_type+".csv"), separator=",", append=False),
                    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(currSavePath,model_type+'_batchSize_'+str(batchSizes)+"_opt_"+opt_type+'_'+'model.{epoch:02d}.h5'),
                                                       save_best_only=True,
                                                       monitor="val_accuracy",
                                                       mode='max',
                                                       save_weights_only=False),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        min_delta=0,
                        patience=10,
                        verbose=0,
                        mode="max",
                        baseline=None,
                        restore_best_weights=False,)]
    opt_func = getattr(tensorflow_addons.optimizers,optimisers[opt_count])
    opt =  AdamW(learning_rate=learning_rate,weight_decay=weight_decay)
    #opt = SGD()
    new_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])   
    new_model.fit(train_generator, validation_data = valid_generator,epochs=num_epochs, 
                                callbacks=csv_logger, steps_per_epoch = train_generator.n//train_generator.batch_size,
                                validation_steps=valid_generator.n//valid_generator.batch_size)
    


# Source path of the validation and test images
src_path_test = "/home/bc/Documents/Imran/Imran_DFU_Project/DFUC2021_test/test"
src_path_val = "/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/val"


# loading and scalling the test images
test_datagen = ImageDataGenerator(rescale=1 / 255.0,fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    color_mode="rgb",
    target_size = (72,72),
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
    target_size = (72,72),
    class_mode="categorical",
    shuffle=True,
    seed=42
) 



# loading the model and generating predicted probabilites of each of the classes
model_test = load_model(r"/home/bc/Documents/Imran/Imran_DFU_Project/DFU_80/dfu_80_vit_results/InceptionResNetV2/AdamW/32/InceptionResNetV2_batchSize_0_opt_AdamW_model.12.h5", 
                        custom_objects={'Patches':Patches,
                                        'PatchEncoder':PatchEncoder})

# evaluating the model on the validation data
model_test.evaluate_generator(valid_generator, steps=valid_generator.n//valid_generator.batch_size)
test_generator.reset()

# making predictions on the test images
pred_class = model_test.predict_generator(test_generator,steps=test_generator.n//test_generator.batch_size)
#cl = np.round(pred_class)



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
result.to_csv("Incp_vit_final.csv",index=False)


























