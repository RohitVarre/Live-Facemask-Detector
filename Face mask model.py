#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.preprocessing import LabelEncoder


# In[3]:


mobile=tf.keras.applications.mobilenet.MobileNet()
model=Sequential()
for layer in mobile.layers[:-6]:
    model.add(layer)
for layer in model.layers:
    layer.trainable=False
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=2,activation='softmax'))
model.summary()


# In[4]:


x=[]
y=[]
labels=[0,1]
import os
data_path=r'C:\Users\Rohit Varre\Desktop\mask data'
categories=os.listdir(data_path)
code=dict(zip(categories,labels))
for folder in categories:
    folder_path=os.path.join(data_path,folder)
    images=os.listdir(folder_path)
    for imge in images:
        image_path=os.path.join(folder_path,imge)
        img=cv2.imread(image_path)
        resized=cv2.resize(img,(224,224))
        x.append(resized)
        y.append(code[folder])
x=np.array(x)
x=x/255
y=np.array(y)
y=np.reshape(y,(-1,1))
print(x.shape)


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
print(y_test)


# In[31]:


model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=5,verbose=2,validation_split=0.2,batch_size=20)


# In[32]:


predictions=model.predict(x=x_test,verbose=0)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=y_test,y_pred=predictions.argmax(axis=1))
print(cm)


# In[33]:


face_clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
label_code={1:'Mask',0:'No Mask'}
color_code={1:(0,255,0),0:(0,0,255)}
print(face_clf)


# In[34]:


while True:
    ret,image=source.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_clf.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        face_img=image[y:y+h,x:x+w]
        img=cv2.resize(face_img,(224,224))
        normal=img/255
        final=np.reshape(normal,(1,224,224,3))
        result=model.predict(final)
        label=np.argmax(result,axis=1)[0]
        
        cv2.rectangle(image,(x,y),(x+w,y+h),color_code[label],2)
        cv2.putText(image,label_code[label],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1.0,color_code[label],2)
        
    cv2.imshow('LIVE',image)
    key=cv2.waitKey(1)
    if key==27:
        break

cv2.destroyAllWindows()
source.release()


# In[ ]:





# In[ ]:




