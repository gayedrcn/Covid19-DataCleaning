#!/usr/bin/env python
# coding: utf-8

# #### Loading Data

# In[3]:


# import required libraries for data visualization and analysis
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[4]:


# Load data from file "adult.csv" as dataframe
data = pd.read_csv("adult.csv")
data.head()


# #### Exploring Data Structure

# In[5]:


# Show summary of records. Print names of columns and their datatypes
data.info()


# In[5]:


# Check your data looking with first 5 rows of table. Print the size of data  
data_prepared=data.head(5)
print(data_prepared)


# #### Exploring Data Statictically

# In[6]:


# Get statistical summary of data
print(data.shape)
data.describe()


# In[7]:


# Find mean values of the first 10 records in dataframe
data.head(10).mean()


# #### Filtering Data 

# In[8]:


# Get records whose "age" is greater and equal than 25
data_prepared = data["age"] >= 25
data[data_prepared]


# In[9]:


# Obtain a subspace of data with "Age","Gender" and "income" features
data_prepared = data[["age","gender","income"]]
print(data_prepared)


# #### Data Cleaning

# In[10]:


# Check duplicate records
print(data.duplicated())


# In[11]:


# Convert categorical "martial-status" data to binary attibutes
data_prepared = pd.get_dummies(data['marital-status'])
data_prepared.head()


# In[12]:


# Check missing values in your data 
data.isnull().sum()


# In[13]:


# Obtain a subset of data with nonmising values
print (data.notnull())
data.notnull().sum()


# In[20]:


# Fill missing values using any technique (mentioned in the lecture) according to data type 
data=data.replace(0, np.nan)
data=data.fillna(method="bfill")
data=data.replace('?', np.nan)
data=data.fillna(method="bfill")
data=data.dropna()
print(data)


# #### Visualizing Data

# 1.Univariate analysis

# In[21]:


# "Age" distibution using histogram, pdf and cdf (using matplotlib  and seaborn)
sns.set_style("whitegrid")
x, y = plt.subplots(1, 2, figsize=(15,5))
y[0].set_ylabel("PDF")
y[1].set_ylabel("CDF")
sns.histplot(data=data["age"], ax = y[0])
sns.histplot(data=data["age"], ax = y[1])


# In[22]:


# Count plot for categorical attibute "education","martial-status" and "occupation" (using seaborn)
sns.set_style("whitegrid")
data_pr = data[["education","marital-status","occupation"]]
sns.set(rc={"figure.figsize":(15,5)})
x = sns.countplot(x="education",data= data_pr)
plt.show()
x = sns.countplot(x="marital-status",data= data_pr)
plt.show()
x = sns.countplot(x="occupation",data= data_pr)
plt.show()


# In[23]:


# Draw violin plot for "educational-num" feature
sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(15,5)})
x = sns.violinplot(x=data["educational-num"])
plt.show()


# 2.Bivariate analysis

# In[24]:


# Draw Seaborn scatter plot of "age" and "fnlwgt" features (colored according to classes (hint:hue))
sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(15,5)})
sns.scatterplot(data=data, x= "age", y ="fnlwgt", hue = "workclass")
plt.show()


# 3.Multivariate analysis

# In[25]:


# Calculate correlation of features and show using heatmap of Seaborn. 
sns.set(rc={"figure.figsize":(15,5)})
sns.heatmap(data.corr())


#  Try to interpret plot

# In[26]:


# Draw seaborn pair plot demonstrating class info.
sns.pairplot(data=data)


#  Try to interpret plot

# ##### Question: Which plots are suitable for categorical data? Which for numerical?

# Plots used for categorical data; It can be given as bar graph, pie chart, stacked graph, scatter plot, heatmap. Plots used for numerical data; Histogram can be given as frequency polygon, line graph, stem and leaf plot, cumulative frequency polynomial, box and whisker plot, mean Standard deviation graph, count plot.

# Note: Homeworks must be named as "YourStudentId_hw1_Ceng489_20Fall".*
