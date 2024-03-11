#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().system('pip install opencv-python ')


# In[35]:


import cv2


# In[36]:


img = cv2.imread('../../d1.jpg ')


# In[37]:


img.shape


# In[38]:


img[0]


# In[39]:


img


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


plt.imshow(img)


# In[42]:


import cv2
while True:
    cv2.imshow('result',img)
    #27 - ASCII of Escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[43]:


haar_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[44]:


haar_data.detectMultiScale(img)


# In[45]:


#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_tickness


# In[46]:


import cv2
while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
    cv2.imshow('result',img)
    #27 - ASCII of Escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[48]:


capture = cv2.VideoCapture(0)
data  = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv2.imshow('result',img)
        #27 - ASCII of Escape
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break
            
capture.release()
cv2.destroyAllWindows()


# In[19]:


import numpy as np
np.save('without_mask.npy',data)


# In[49]:


np.save('with_mask.npy',data)


# In[21]:


plt.imshow(data[0])


# In[ ]:





# In[ ]:




