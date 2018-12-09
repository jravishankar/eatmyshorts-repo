"""
Eat My Shorts
A Simpson's Image Classifier

Joshua Ravishankar
Bryan Diaz
Guanghan Pan
"""

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
characters = {"homer":0,"bart":1,"marge":2,"principal":3,"lisa":4}
label=[]
data1=[]
imgs = []
counter = 0
pic_size = 32

#Homer Simpson's Folder (0 is label for Homer Simpson)
for file in os.listdir(path_train):
    if file.endswith(".DS_Store"):
        continue
    imgs.append(file)
shuffle(imgs)

# Build our list of labels by reading in the images
for f in imgs:
    image_data=cv2.imread(os.path.join(path_train + "/",f), cv2.IMREAD_COLOR)
    #print(str(f))
    #print(image_data)
    image_data=cv2.resize(image_data,(pic_size,pic_size))
    if f.startswith("homer"):
        label.append(0)
    elif f.startswith("bart"):
        label.append(1)
    elif f.startswith("marge"):
        label.append(2)
    elif f.startswith("principal"):
        label.append(3)
    elif f.startswith("lisa"):
        label.append(4)
    try:
        data1.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")

### Building the CNN
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout

model=Sequential()
model.add(Conv2D(kernel_size=(5,5),filters=6,input_shape=(pic_size,pic_size,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(5,5),filters=16,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
#model.add(Dropout(0.3))
model.add(Dense(120,activation="relu"))
model.add(Dense(84,activation="relu"))
model.add(Dense(5,activation="softmax"))
model.summary()
model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])



data1=np.array(data1)
data1=data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],3) ##
for i in range(len(label)):
    label[i] = label[i] == np.array([0,1,2,3,4])
#print(label)



labels=np.array(label)

print (data1.shape)
print (labels.shape)

model.fit(data1,labels,validation_split=0.1,epochs=20,batch_size=10)
model.save("model.h5")

# Read in and label test data
test_data=[]
id=[]
counter=0
for file in os.listdir(path_test):
    image_data=cv2.imread(os.path.join(path_test,file), cv2.IMREAD_COLOR)
    try:
        image_data=cv2.resize(image_data,(pic_size,pic_size))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("bo")
    counter+=1
    if counter%100==0:
        print (counter," image data retreived")

test_data1=np.array(test_data)
print (test_data1.shape)
test_data1=test_data1.reshape((test_data1.shape)[0],(test_data1.shape)[1],(test_data1.shape)[2],3)

# Takes the maximum value from the softmax output and defines that as predicted label
dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data1)
predicted_labels=np.round(predicted_labels,decimals=2)
labels = []
for arr in predicted_labels:
    max = 0
    maxInd = 0
    for pos in range(len(arr)):
        if arr[pos] > max:
            max = arr[pos]
            maxInd = pos
    labels.append(maxInd)


dataframe_output["label"]=labels
dataframe_output.to_csv("submission.csv",index=False)


# Check test accuracy by comparing predicted with correfy
with open('submission.csv') as file:
    reader = csv.DictReader(file)
    total = 0
    corr = 0
    for row in reader:
        id = row['id']
        label = row['label']
        #print(id[4:])
        #print(label)
        if id.startswith("homer") and label == "0":
            corr += 1
        elif id.startswith("bart") and label == "1":
            corr += 1
        elif id.startswith("marge") and label == "2":
            corr += 1
        elif id.startswith("principal") and label == "3":
            corr += 1
        elif id.startswith("lisa") and label == "4":
            corr += 1
        total += 1


print("Test Accuracy: " + str((corr/total) * 100))
