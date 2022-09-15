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


def acc_plotter(x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9, lbl1, lbl2, lbl3, lbl4, lbl5, lbl6, lbl7, lbl8, lbl9):
    '''
        Helper function to plot the Validation Accuracies
    '''
    plt.figure(figsize=(35,18))
    a = plt.subplot(2,2,1)
    a.plot(x1,y1,label=lbl1)
    a.legend(loc='lower right')
    a.plot(x2,y2,label=lbl2)
    a.legend(loc='lower right')
    a.plot(x3,y3,label=lbl3)
    a.legend(loc='lower right')
    a.plot(x4,y4,label=lbl4)
    a.legend(loc='lower right')
    a.plot(x5,y5,label=lbl5)
    a.legend(loc='lower right')
    a.plot(x6,y6,label=lbl6)
    a.legend(loc='lower right')
    a.plot(x7,y7,label=lbl7)
    a.legend(loc='lower right')
    a.plot(x8,y8,label=lbl8)
    a.legend(loc='lower right')
    a.plot(x9,y9,label=lbl9)
    a.legend(loc='lower right')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validatin Accuracy")
    
def loss_plotter(x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9, lbl1, lbl2, lbl3, lbl4, lbl5, lbl6, lbl7, lbl8, lbl9):
    '''
        Helper Function to plot the validation losses
    '''
    plt.figure(figsize=(35,18))
    a = plt.subplot(2,2,1)
    a.plot(x1,y1,label=lbl1)
    a.legend(loc='upper right')
    a.plot(x2,y2,label=lbl2)
    a.legend(loc='upper right')
    a.plot(x3,y3,label=lbl3)
    a.legend(loc='upper right')
    a.plot(x4,y4,label=lbl4)
    a.legend(loc='upper right')
    a.plot(x5,y5,label=lbl5)
    a.legend(loc='upper right')
    a.plot(x6,y6,label=lbl6)
    a.legend(loc='upper right')
    a.plot(x7,y7,label=lbl7)
    a.legend(loc='upper right')
    a.plot(x8,y8,label=lbl8)
    a.legend(loc='upper right')
    a.plot(x9,y9,label=lbl9)
    a.legend(loc='upper right')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validatin Loss")
    
    
# This part is for the full dataset
model1 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/IncpFull.csv')
model2 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp60.csv')
model3 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp65.csv')
model4 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp70.csv')
model5 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp75.csv')
model6 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp80.csv')
model7 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp85.csv')
model8 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp90.csv')
model9 = pd.read_csv('D:\MSc Studies\MSc Project\MSc Work\MS_Project\Model Results\Results Visulisation\All Model Ressults/Incp95.csv')

# number of epochs for the top models
x1 = model1['epoch']
x2 = model2['epoch']
x3 = model3['epoch']
x4 = model4['epoch']
x5 = model5['epoch']
x6 = model6['epoch']
x7 = model7['epoch']
x8 = model8['epoch']
x9 = model9['epoch']

# val accuracies
y1 = model1['val_accuracy']
y2 = model2['val_accuracy']
y3 = model3['val_accuracy']
y4 = model4['val_accuracy']
y5 = model5['val_accuracy']
y6 = model6['val_accuracy']
y7 = model7['val_accuracy']
y8 = model8['val_accuracy']
y9 = model9['val_accuracy']

# plotting the graph
acc_plotter(x1, x2, x3, x4, x5, x6, x7, x8, x9,
            y1, y2, y3, y4, y5, y6, y7, y8, y9,
            'InceptionResNetV2_Full', 'InceptionResNetV2_60', 'InceptionResNetV2_65',
            'InceptionResNetV2_70', 'InceptionResNetV2_75', 'InceptionResNetV2_80',
            'InceptionResNetV2_85', 'InceptionResNetV2_90', 'InceptionResNetV2_95')

# validation losses
# val accuracies
y1 = model1['val_loss']
y2 = model2['val_loss']
y3 = model3['val_loss']
y4 = model4['val_loss']
y5 = model5['val_loss']
y6 = model6['val_loss']
y7 = model7['val_loss']
y8 = model8['val_loss']
y9 = model9['val_loss']

# plotting the graph
loss_plotter(x1, x2, x3, x4, x5, x6, x7, x8, x9,
            y1, y2, y3, y4, y5, y6, y7, y8, y9,
            'InceptionResNetV2_Full', 'InceptionResNetV2_60', 'InceptionResNetV2_65',
            'InceptionResNetV2_70', 'InceptionResNetV2_75', 'InceptionResNetV2_80',
            'InceptionResNetV2_85', 'InceptionResNetV2_90', 'InceptionResNetV2_95')





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

































