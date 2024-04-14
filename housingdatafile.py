#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is a basic housing dataset. With it I will showcase Data Cleaning and Feature Engineering, Descriptive Statistics and Machine Learning.


# In[2]:


#Firstly, we will import pandas and numpy libraries


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


#Now we will import our dataset


# In[5]:


inpPath = 'C:/Users/jammy/OneDrive/Desktop/'
housingdata = pd.read_csv(inpPath + 'housing.csv', delimiter=',')


# In[6]:


#Now we will have a look at the data


# In[7]:


housingdata


# In[8]:


#We can see that the formatting of the dates is a bit messy, so we will try and change this


# In[9]:


from datetime import datetime


# In[10]:


housingdata['date'] = pd.to_datetime(housingdata['date'], errors='coerce')
housingdata['date'] = housingdata['date'].dt.strftime('%Y-%m-%d')
print(housingdata)


# In[11]:


#For the purpose of the dataset, lets remove the waterfront column as while 163 houses have waterfronts, I don't feel it is important in buying a house.


# In[12]:


housingdata = housingdata.drop(columns='waterfront')


# In[13]:


print(housingdata.columns)


# In[14]:


#Now we will clean the data, addressing any null values, duplicates and outliers that may exist


# In[15]:


#First lets check for and remove any null values, if there is any - we there aren't


# In[16]:


print(housingdata.isnull().sum()) 


# In[17]:


#Next we will look for any duplicates in the id section and remove them


# In[18]:


housingdata = housingdata.drop_duplicates(subset=['id'], keep='first')
print(housingdata)


# In[19]:


#Finally we will use IQR on the price column to remove any outliers - firstly establishing Q1 and Q3 values then subtracting to establish the IQR


# In[20]:


Q1 = housingdata['price'].quantile(0.25)
Q3 = housingdata['price'].quantile(0.75)
IQR = Q3-Q1
print(IQR)


# In[21]:


#Next we establish to upper and lower bounds


# In[22]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(lower_bound)
print(upper_bound)


# In[23]:


#Now I filter out whatever is outside the lower and upper bounds, these are the outliers.


# In[24]:


housingdata_filtered = housingdata[(housingdata['price'] >= lower_bound) & (housingdata['price'] <= upper_bound)]


# In[25]:


#How many entries were removed as outliers?


# In[26]:


print(len(housingdata) - len(housingdata_filtered))


# In[27]:


#We now have our cleaned dataset


# In[28]:


print(housingdata_filtered)


# In[29]:


#Firstly, lets have a look at some descriptive statistics


# In[30]:


#Let's have a look at the different data types


# In[31]:


data_type = housingdata_filtered.dtypes
print(data_type)


# In[32]:


#Let's get an idea of the shape of the data


# In[33]:


housingdata_filtered.shape


# In[34]:


#Let's have a look at the describing some of the numeric columns


# In[35]:


#Overview of Numeric values
housingdata_filtered.describe(include=np.number)


# In[36]:


#I notice I am getting scientific notations "e+" here so I need to address this to get a better picture of the descriptive statistics


# In[37]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[38]:


housingdata_filtered.describe(include=np.number)


# In[39]:


#Let's do some more - for example let's find the mode, variance, and kurtosis of the housing prices


# In[40]:


print("Kurtosis:" + " " + str(housingdata_filtered["price"].kurtosis()))


# In[41]:


#This shows the data is normally distrubuted, and has little amount of outliers


# In[42]:


print("Mode:" + str(housingdata_filtered["price"].mode()))


# In[43]:


print("Variance:" + " " + str(housingdata_filtered["price"].var()))


# In[44]:


#This shows that the prices are widely spread out and varied


# In[45]:


#Now let's look at some aspects of feature engineering 


# In[46]:


#Let's create a function that allows us to calculate the price per floor in each house


# In[47]:


housingdata_filtered['price_per_floor'] = housingdata_filtered['price'] / housingdata_filtered['floors']
housingdata_filtered


# In[48]:


#Towards the last column of the dataset, we can see a new column that reflects the price per floor for us.


# In[49]:


#Let's apply a dollar sign to the price and price per floor columns to make it more readable


# In[50]:


housingdata_filtered['price'] = '$' + housingdata_filtered['price'].astype(str)
housingdata_filtered['price_per_floor'] = '$' + housingdata_filtered['price_per_floor'].astype(str)
housingdata_filtered


# In[51]:


#We can see that the condition column is numeric, let's use label encoding to change this


# In[52]:


#First, lets find the unique values


# In[53]:


housingdata_filtered['condition'].unique()


# In[54]:


#Now let's create an encoding dictionary to help us label these columns


# In[55]:


encoding_dict = {1: 'Poor', 2: 'Unsatisfactory', 3: 'Okay', 4: 'Good', 5: 'Great', np.nan:np.nan}
housingdata_filtered.loc[:, 'condition_encoded'] = housingdata_filtered['condition'].map(encoding_dict)
housingdata_filtered


# In[56]:


#Now we can see that there is a new column at end called "condition_encoded" with our new values


# In[57]:


#Now to finish lets quickly look at some machine learning techniques.


# In[58]:


#For Supervised Machine Learning let's first look at K-Nearest Neighbhour, let's import some libraries for this


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[60]:


#Now let's split the data


# In[61]:


Xdf = housingdata_filtered[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
Ydf = y = housingdata_filtered['price']


# In[62]:


#Now let's split between train and test set


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(Xdf, Ydf, test_size=0.3, random_state=0)


# In[64]:


#Lets look at our train and test sets to see how to split is broken up


# In[65]:


print('Train set')
print(X_train.shape)
print(y_train.shape)


# In[66]:


print('Test set')
print(X_test.shape)
print(y_test.shape)


# In[67]:


#Let's set the algorithm parameters


# In[68]:


clf = KNeighborsClassifier(n_neighbors=3)


# In[69]:


#Let's fit the data


# In[70]:


clf.fit(X_train, y_train)


# In[71]:


#Let's look at the output's score


# In[72]:


print(clf.score(X_test, y_test))


# In[73]:


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[74]:


#Lets also implement Naive Bayes Algorithm


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[76]:


#Again we split the data


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(Xdf, Ydf, test_size=0.3, random_state=0)


# In[78]:


#Set the algorithm parameters


# In[79]:


clf = GaussianNB(priors=None)


# In[80]:


#Fit the Data


# In[81]:


clf.fit(X_train, y_train)


# In[82]:


#Analyse the scores


# In[83]:


print(clf.score(X_test, y_test))


# In[84]:


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[85]:


#For Unsupervised Machine learning let's look at the Kmeans Clustering Technique


# In[86]:


#Let's first import the necessary libraries


# In[87]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[88]:


#First let's pick the columns to cluster, lets cluster based on the features of the house


# In[89]:


Xdf = housingdata_filtered[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]


# In[90]:


#Now we perform clustering for K up to 15, create the inertia lis


# In[91]:


inertiaLst = []
kVal = 1

while kVal <= 15:
    kmeans = KMeans(n_clusters=kVal, n_init=10)
    kmeans.fit(Xdf)
    inertiaLst.append([kVal, kmeans.inertia_])
    kVal += 1
    
print(inertiaLst)


# In[92]:


#Now we will plot this list


# In[93]:


inertiaArr = np.array(inertiaLst).transpose()
plt.plot(inertiaArr[0], inertiaArr[1])
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


# In[94]:


#Based off this, we can assume the elbow point for the dataset is when the kval=4 or 5

