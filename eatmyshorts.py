# Suck my donkey dick


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import shuffle
import csv


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../FinalProject"))

# Any results you write to the current directory are saved as output.

import cv2
path_train ="../FinalProject/train"
path_test ="../FinalProject/test/test"
label=[]
data1=[]
imgs = []
counter = 0

#Homer Simpson's Folder (0 is label for Homer Simpson)
for file in os.listdir(path_train):
    if file.endswith(".DS_Store"):
        continue
    imgs.append(file)
shuffle(imgs)

for f in imgs:
    image_data=cv2.imread(os.path.join(path_train + "/",f), cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(96,96))
    if f.startswith("hpic"):
        label.append(0)
    elif f.startswith("bpic"):
        label.append(1)
    try:
        data1.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%100==0:
        print (counter," image data retreived")

"""
#Bart Simpson's Folder (1 is label for Bart Simpson)
for file in os.listdir(path_train + "/bart_simpson"):
    if file.endswith(".DS_Store"):
        continue
    image_data=cv2.imread(os.path.join(path_train + "/homer_simpson",file), cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(96,96))
    label.append(1)
    try:
        data1.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%100==0:
        print (counter," image data retreived")
"""
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout

model=Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(96,96,1),activation="relu"))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu"))
model.add(Conv2D(kernel_size=(5,5),filters=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])


data1=np.array(data1)
data1=data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],1)
labels=np.array(label)

print (data1.shape)
print (labels.shape)

model.fit(data1,labels,validation_split=0.35,epochs=10,batch_size=10)
model.save_weights("model.h5")

test_data=[]
id=[]
counter=0
for file in os.listdir(path_test):
    image_data=cv2.imread(os.path.join(path_test,file), cv2.IMREAD_GRAYSCALE)
    try:
        image_data=cv2.resize(image_data,(96,96))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("bo")
    counter+=1
    if counter%100==0:
        print (counter," image data retreived")

test_data1=np.array(test_data)
print (test_data1.shape)
test_data1=test_data1.reshape((test_data1.shape)[0],(test_data1.shape)[1],(test_data1.shape)[2],1)
dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data1)
predicted_labels=np.round(predicted_labels,decimals=2)
labels=[1 if value>0.5 else 0 for value in predicted_labels]
dataframe_output["label"]=predicted_labels
dataframe_output.to_csv("submission.csv",index=False)



with open('submission.csv') as file:
    reader = csv.DictReader(file)
    total = 0
    corr = 0
    for row in reader:
        id = row['id']
        label = row['label']
        #print(id[4:])
        #print(label)
        if int(id[4:]) >= 1342 and float(label) < 0.50:
            corr += 1
        elif int(id[4:]) < 1342 and float(label) >= 0.50:
            corr += 1
        total += 1


print("Test Accuracy: " + str((corr/total) * 100))
