#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


# In[3]:


A = pd.read_csv("C:/Users/barsh/Downloads/Etlhive/Life Expectancy Data.csv")


# In[4]:


A.head()


# In[5]:


A.describe()


# In[6]:


A.info()


# In[ ]:





# In[7]:


A.nunique()


# In[8]:


A.shape


# In[9]:


A.isna().sum()


# In[10]:


for i in A.columns:
    if A[i].dtypes == "object":
        t=A[i].mode()[0]
        A[i]=A[i].fillna(t)
    else:
        t=A[i].mean()
        A[i]=A[i].fillna(t)


# In[11]:


A.isna().sum()


# In[12]:


cat=[]
con=[]
for i in A.columns:
    if A[i].dtypes == "object":
        cat.append(i)
    else:
        con.append(i)


# In[13]:


cat


# In[14]:


con


# In[15]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
from pandas import DataFrame
A1=pd.DataFrame(ss.fit_transform(A[con]),columns=con)


# In[16]:


A1


# In[17]:


A.shape


# In[18]:


A1[(A1['Life expectancy ']<-3)|(A1['Life expectancy ']>3)]   


# In[19]:


outliers=[]
for i in A1.columns:
    outliers.extend(A1[(A1[i]<-3)|(A1[i]>3)].index)
    


# In[20]:


outliers


# In[21]:


len(outliers)


# In[22]:


Y=A[["Life expectancy "]]
X=A.drop(labels="Life expectancy ",axis=1)


# In[23]:


cat=[]
con=[]
for i in X.columns:
    if A[i].dtypes=="object":
        cat.append(i)
    else:
        con.append(i)


# In[24]:


cat


# In[25]:


con


# In[26]:


con


# In[27]:


A.corr()["Life expectancy "].sort_values


# In[28]:


Q=A.corr()["Life expectancy "]
Q=Q[(Q<-0.5)|(Q>0.5)]
Q=Q[Q!=1]
Q.index


# In[ ]:





# In[29]:


plt.figure(figsize=(12,12))
x=1
for i in Q.index:
    if (A[i].dtype=="object"):
        plt.subplot(6,5,x)
        sb.boxplot(A[i],A["Life expectancy "])
        x=x+1
    else:
        plt.subplot(6,5,x)
        sb.scatterplot(A[i],A["Life expectancy "])
        x=x+1


# In[30]:


Y=A[["Life expectancy "]]
X=A[Q.index]


# In[31]:


X.head()


# In[32]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xnew=pd.DataFrame(ss.fit_transform(X),columns=Q.index)


# In[33]:


Xnew


# In[34]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# In[35]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model=lm.fit(xtrain,ytrain)


# In[40]:


tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)


# In[41]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(ytrain,tr_pred)


# In[42]:


mean_absolute_error(ytest,ts_pred)


# In[ ]:




