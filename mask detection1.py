#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')


# In[3]:


with_mask.shape


# In[4]:


without_mask.shape


# In[5]:


with_mask = with_mask.reshape(200,50 * 50 * 3)
without_mask = without_mask.reshape(200,50 * 50 * 3)


# In[6]:


with_mask.shape
without_mask.shape


# In[7]:


X = np.r_[with_mask,without_mask]


# In[8]:


X.shape


# In[9]:


labels = np.zeros(X.shape[0])


# In[10]:


labels[200:] = 1.0


# In[11]:


name = {0: 'Mask', 1: 'No Mask'}


# In[12]:


# svm - Support Vector Machine
# SVC - Support Vector Classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)


# In[15]:


x_train.shape


# In[16]:


# PCA - Principle Component Analysis
from sklearn.decomposition import PCA


# In[17]:


pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)


# In[18]:


x_train[0]


# In[19]:


x_train.shape


# In[20]:



x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)


# In[21]:


svm = SVC()
svm.fit(x_train, y_train)


# In[22]:


#x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[23]:


accuracy_score(y_test, y_pred)


# In[24]:


haar_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            face = face.reshape(1,-1)
            #face = pca.transform(face)
            pred = svm.predict(face)
            n = name[int(pred)]
            cv2.putText(img, n ,(x,y), font, 1, (244,250,250), 2)
            print(n)
        cv2.imshow('result',img)
        #27 - ASCII of Escape
        if cv2.waitKey(2) == 27:
            break
            
capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




