#!/usr/bin/env python
# coding: utf-8

# ## Neccessary Libraries Required For Data Analysis:
# 
#  * Numpy
#  * Matplotlib
#  * seaborn
#  * Pandas

# In[1]:


# importing neccessary libraries for data analysis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ## Importing the dataset for data analysis

# In[2]:


# importing the data from the dataset and storing in a dataframe

df = pd.read_csv('laptop_prices.csv')


# In[3]:


# Viewing first 5 rows from the dataframe

df.head()


# ## Uni-Variate Analysis

# In[4]:


# viewing the shape i.e no. of rows and columns of the dataframe

df.shape


# In[5]:


# viewing information about data in each column

df.info()


# In[6]:


# checking for duplicated rows(if any) present in df

df.duplicated().sum()


# In[7]:


# checking for missing(or null) values(if any) in df

df.isnull().sum()


# In[8]:


# describing the statistical information about the numeric data

df.describe()


# In[9]:


# plotting the distribution of price

sns.histplot(x = df['Price'], kde = True)
plt.show()


# ### Observations from above analysis:
# 
# * presence of no duplicate rows in the dataset
# * presence of no missing values in the dataset
# * 'Unnamed: 0' column is of no use, so better to remove
# * 'Ram' column can be converted to int type by removing 'GB'
# * 'Weight' column can be converted to float type by removing 'kg'
# * There are many pieces of information stored in columns 'ScreenResolution', 'Cpu', 'Memory', 'Gpu' which can be extracted to check if there exists any relationship with dependant variable 'price'
# * The distribution of data in price column is left skewed and needs transformation
# 

# ## Data Preprocessing and cleaning

# In[10]:


# removing 'Unnamed: 0' column from df

df.drop( columns = ['Unnamed: 0'], inplace = True)
df.head()


# In[11]:


# removing 'GB' and 'kg' from 'Ram' and 'Weight' and converting to int and foat respectively

df['Ram'] = df['Ram'].apply(lambda x : int(x.replace('GB','')))
df['Weight'] = df['Weight'].apply(lambda x : float(x.replace('kg','')))
df.head()


# In[12]:


df.info()


# In[13]:


# counting different categories of values present in 'ScreenResolution' column

df['ScreenResolution'].value_counts()


# In[14]:


# plotting the distribution of weight

sns.histplot(x = df['Weight'], kde = True)
plt.show()


# In[15]:


# extracting whether ips display present or not and then creating a new column

df['Ips_Panel'] = df['ScreenResolution'].apply(lambda x : 1 if 'IPS' in x else 0)
df.sample(5)


# In[16]:


# extracting whether touchscreen present or not and then creating a new column

df['Touchscreen'] = df['ScreenResolution'].apply(lambda x : 1 if 'Touchscreen' in x else 0)
df.sample(5)


# In[17]:


# extracting resolution info

# extracting Width in pixel for each laptop and storing in new column
df['Width'] = df['ScreenResolution'].apply(lambda x : int((x.split('x')[0]).split()[-1]))

# extracting Height in pixel for each laptop and storing in a new column
df['Height'] = df['ScreenResolution'].apply(lambda x : int(x.split('x')[1]))


# In[18]:


# removing 'ScreenResolution' column

df.drop(columns = ['ScreenResolution'], inplace = True)

#viewing the dataframe after deletion

df.head()


# In[19]:


# counting different categories of values present in 'Cpu' column

df['Cpu'].value_counts()


# In[20]:


# Extracting Cpu brand and creating a new column to store it

def extract_cpu(cpu):
    processor = " ".join( cpu.split()[0 : 3] )
    if processor == 'Intel Core i7' or processor == 'Intel Core i5' or processor == 'Intel Core i3':
        return processor
    elif processor.split()[0] == 'Intel':
        return "Other Intel Processor"
    else:
        return processor.split()[0]

# creating a new column to store cpu brands by applying extract_cpu() on 'Cpu' column
 
df['Cpu_Brand'] = df['Cpu'].apply(extract_cpu)


# In[21]:


# viewing the dataframe

df.sample(5)


# In[22]:


# removing cpu column 

df.drop( columns = ['Cpu'], inplace = True)

# viewing dataframe after removal of column

df.head()


# In[23]:


# Checking diffrent categories of data present in the newly created column

df['Cpu_Brand'].value_counts()


# In[24]:


# removing the row having 'Samsung' as the type of 'Cpu_Brand'

df = df[ df['Cpu_Brand'] != 'Samsung']

# viewing the categories of data in 'Cpu_Brand' after the removal

df['Cpu_Brand'].value_counts()


# In[25]:


df.head()


# In[26]:


# plotting the count of each category of Cpu_Brand

df['Cpu_Brand'].value_counts().plot(kind='bar')
plt.xlabel('Cpu Brand')
plt.ylabel('Count')
plt.show()


# In[27]:


# counting different categories of values present in 'Memory' column

df['Memory'].value_counts()


# In[28]:


# Extracting various memory types from the 'Memory' column

# function for extracting the HDD memory types
def extract_hdd(mem):
    hdd_size = 0
    words = mem.split()
    for i in range(0, len(words)):
        
        # extracting memory size for 'HDD' category
        if words[i] == "HDD":
            if "TB" in words[i-1]:
                val = words[i-1].replace("TB","")
                if val == "1.0":
                    hdd_size = 1 * 1024
                else:
                    hdd_size = int(val) * 1024
            else:
                hdd_size = int(words[i-1].replace("GB",""))

    return hdd_size

# function for extracting the SSD memory types
def extract_ssd(mem):
    ssd_size = 0
    words = mem.split()
    for i in range(0, len(words)):
        
        # extracting memory size for 'SSD' category
        if words[i] == "SSD":
            if "TB" in words[i-1]:
                ssd_size = int(words[i-1].replace("TB","")) * 1024
            else:
                ssd_size = int(words[i-1].replace("GB",""))

    return ssd_size

# function for extracting the Flash Storage memory types
def extract_flash(mem):
    flash_size = 0
    words = mem.split()
    for i in range(0, len(words)):
        
        # extracting memory size for 'Flash Storage' category
        if words[i] == "Flash":
            flash_size = int(words[i-1].replace("GB",""))

    return flash_size

# function for extracting the Hybrid memory types
def extract_hybrid(mem):
    hybrid_size = 0
    words = mem.split()
    for i in range(0, len(words)):
        
        # extracting memory size for 'HDD' category
        if words[i] == "Hybrid":
            if "TB" in words[i-1]:
                val = words[i-1].replace("TB","")
                if val == "1.0":
                    hybrid_size = 1 * 1024
                else:
                    hybrid_size = int(val) * 1024
            else:
                hybrid_size = int(words[i-1].replace("GB",""))

    return hybrid_size


# In[29]:


# applying the extractor functions on memory column and storing the results into new columns

df['Hdd'] = df['Memory'].apply(extract_hdd)
df['Ssd'] = df['Memory'].apply(extract_ssd)
df['Flash_Storage'] = df['Memory'].apply(extract_flash)
df['Hybrid'] = df['Memory'].apply(extract_hybrid)


# In[30]:


# viewing the dataframe
df.sample(10)


# In[31]:


# removing memory column after extraction of memory types

df.drop(columns = 'Memory', inplace = True)


# In[32]:


# viewing dataframe after removal

df.head()


# In[33]:


# counting different categories of values present in 'Gpu' column

df['Gpu'].value_counts()


# In[34]:


# creating and storing gpu brands against each laptop

df['Gpu_Brand'] = df['Gpu'].apply(lambda x : x.split()[0])


# In[35]:


# Viewing df

df.head()


# In[36]:


# removing Gpu column

df.drop(columns = 'Gpu', inplace = True)


# In[37]:


# Viewing df

df.head()


# In[38]:


# checking counts of each category in 'Gpu_Brand' column

df['Gpu_Brand'].value_counts()


# In[39]:


# plotting the count of each category of Gpu_Brand

df['Gpu_Brand'].value_counts().plot(kind='bar')
plt.xlabel('Gpu Brand')
plt.ylabel('Count')
plt.show()


# In[40]:


# checking counts of each category in 'OpSys' column

df['OpSys'].value_counts()


# In[41]:


# plotting the count of each category of OpSys

df['OpSys'].value_counts().plot(kind='bar')
plt.xlabel('OpSys')
plt.ylabel('Count')
plt.show()


# In[42]:


# function to categorize os

def categorise_os(opsys):
    if opsys == "Windows 10" or opsys == "Windows 7" or opsys == "Windows 10 S":
        return "Windows"
    elif opsys == "macOS" or opsys == "Mac OS X":
        return "Mac"
    else:
        return "Other OS / No OS"


# In[43]:


# applying categorise_os() on 'OpSys' column and storing result in new column

df['Operating_System'] = df['OpSys'].apply(categorise_os)


# In[44]:


# removing 'OpSys' column 

df.drop(columns = 'OpSys', inplace = True)


# In[45]:


# viewing df

df.sample(5)


# In[46]:


# plotting the count of each category of Operating_System

df['Operating_System'].value_counts().plot(kind='bar')
plt.xlabel('Operating System')
plt.ylabel('Count')
plt.show()


# In[47]:


# checking counts of each category in 'TypeName' column

df['TypeName'].value_counts()


# In[48]:


# plotting the count of each category of TypeName

df['TypeName'].value_counts().plot(kind='bar')
plt.xlabel('TypeName')
plt.ylabel('Count')
plt.show()


# In[49]:


# checking counts of each category in 'Company' column

df['Company'].value_counts()


# In[50]:


# plotting the count of each category of Company

df['Company'].value_counts().plot(kind='bar')
plt.xlabel('Company')
plt.ylabel('Count')
plt.show()


# In[51]:


# checking counts of each category in 'Ram' column

df['Ram'].value_counts()


# In[52]:


# removing the row with 64gb ram

df = df[ df['Ram'] != 64]

# viewing df

df.sample(5)


# In[53]:


# plotting the count of each category of ram

df['Ram'].value_counts().plot(kind='bar')
plt.xlabel('Ram')
plt.ylabel('Count')
plt.show()


# In[54]:


# viewing the structure of the df after preprocessing and cleaning the data

df.shape


# In[55]:


# viewing the data information in each column

df.info()


# In[56]:


# checking for missing values

df.isnull().sum()


# In[57]:


# describing the statistical information about the numeric data

df.describe()


# ### Observations
# 
# * Data in 'Price' column is not well organised. Might be due to the presence of outliers in data

# # Bi-Variate Analysis

# In[58]:


# performing bi-variate analysis on display Components of laptops

sns.barplot(x = df['Inches'], y = df['Price'])
plt.xticks(rotation = 90)
plt.show()


# In[59]:


plt.scatter(x = df['Inches'], y = df['Price'])
plt.xlabel('Inches')
plt.ylabel('Prices')
plt.show()


# In[60]:


# finding the correlation coefficient between price and Inches from correlation matrix

df.corr()['Price']['Inches']


# In[61]:


sns.barplot(x = df['Ips_Panel'], y = df['Price'])
plt.show()


# In[62]:


# finding the correlation coefficient between price and Ips panel from correlation matrix

df.corr()['Price']['Ips_Panel']


# In[63]:


sns.barplot(x = df['Touchscreen'], y = df['Price'])
plt.show()


# In[64]:


# finding the correlation coefficient between price and Touchscreen from correlation matrix

df.corr()['Price']['Touchscreen']


# In[65]:


sns.barplot(x = df['Width'], y = df['Price'])
plt.xticks(rotation = 90)
plt.show()


# In[66]:


# finding the correlation coefficient between price and Width from correlation matrix

df.corr()['Price']['Width']


# In[67]:


sns.barplot(x = df['Height'], y = df['Price'])
plt.show()


# In[68]:


# finding the correlation coefficient between price and Height from correlation matrix

df.corr()['Price']['Height']


# ### Observations after bi-variate analysis on display components of laptop
# 
# * There is a very less correlation(since < 0.1, close to 0) between Price and Inches so, Inches column can be dropped
# * Ips Panel and Touchscreen columns have quite moderate correlation with price and can be kept
# * Width and Height have good correlation with price having almost the same correlation coefficient

# In[69]:


# performing bi-variate analysis on company column

sns.barplot(x = df['Company'], y = df['Price'])
plt.xticks(rotation = 90)
plt.show()


# In[70]:


# performing bi-variate analysis on TypeName column

sns.barplot(x = df['TypeName'], y = df['Price'])
plt.xticks(rotation = 45)
plt.show()


# In[71]:


# performing bi-variate analysis on Operating System column

sns.barplot(x = df['Operating_System'], y = df['Price'])
plt.show()


# In[72]:


# performing bi-variate analysis on Weight column

plt.scatter(x = df['Weight'], y = df['Price'])
plt.xlabel('Weight')
plt.ylabel('Price')
plt.show()


# In[73]:


# finding the correlation coefficient between price and Weight from correlation matrix

df.corr()['Price']['Weight']


# ### Observations after bi-variate analysis on company, type, OS and weight of laptops
# 
# * There is a variation of price with company of laptops
# * There is a variation of price with the type of laptops
# * There is a variation of price with the type of OS in laptops
# * There is a moderate correlation between weight and price of laptops

# In[74]:


# performing bi-variate analysis on memory Components of laptops

sns.barplot(x = df['Ram'], y = df['Price'])
plt.show()


# In[75]:


# finding the correlation coefficient between price and Ram from correlation matrix

df.corr()['Price']['Ram']


# In[76]:


sns.barplot(x = df['Hdd'], y = df['Price'])
plt.show()


# In[77]:


# finding the correlation coefficient between price and Hdd from correlation matrix

df.corr()['Price']['Hdd']


# In[78]:


sns.barplot(x = df['Ssd'], y = df['Price'])
plt.show()


# In[79]:


# finding the correlation coefficient between price and Ssd from correlation matrix

df.corr()['Price']['Ssd']


# In[80]:


sns.barplot(x = df['Flash_Storage'], y = df['Price'])
plt.show()


# In[81]:


# finding the correlation coefficient between price and Flash_Storage from correlation matrix

df.corr()['Price']['Flash_Storage']


# In[82]:


sns.barplot(x = df['Hybrid'], y = df['Price'])
plt.show()


# In[83]:


# finding the correlation coefficient between price and Hybrid from correlation matrix

df.corr()['Price']['Hybrid']


# ### Observations after bi-variate analysis on memory components of laptop
# 
# * There is a strong correlation between Ram sizes and Prices of laptops
# * There is a less correlation between Hdd sizes and Prices of laptops
# * There is a strong correlation between Ssd sizes and Prices of laptops
# * There is almost negligible correlation(since close to 0) between Flash Storage sizes and Prices of laptops
# * There is almost negligible correlation(since close to 0) between Hybrid sizes and Prices of laptops
# * Columns 'Flash_Storage' and 'Hybrid' can be dropped
# * From business perspective although hdd is very weakly correlated with price, it can be kept

# In[84]:


# removing columns 'Flash_Storage' and 'Hybrid' 

df.drop(columns = ['Flash_Storage', 'Hybrid'], inplace = True)

# viewing a sample of data from df after removal of columns

df.sample(10)


# In[85]:


# performing bi-variate analysis on Cpu_Brand of laptops

sns.barplot(x = df['Cpu_Brand'], y = df['Price'])
plt.xticks(rotation = 45)
plt.show()


# In[86]:


# performing bi-variate analysis on Gpu_Brand of laptops

sns.barplot(x = df['Gpu_Brand'], y = df['Price'])
plt.xticks(rotation = -45)
plt.show()


# ### Observations after bi-variate analysis on Cpu and Gpu brands of laptop
# 
# * There is a variation of price of laptops with the CPU brands of laptops
# * There is a variation of price of laptops with the GPU brands of laptops

# # Correlation Analysis

# In[87]:


# Displaying the correlation matrix

df.corr()


# In[88]:


# displaying the heatmap for the above correlation matrix for visualization

sns.heatmap(df.corr())
plt.show()


# ### Observations
# 
# * Almost all the variables have a good correlaion with the target column price
# * Inches and Weight have a strong correlation between themselves
# * similarly, width and height have strong correlation between themselves
# * Instead a new column can be introduced which have a relation with Inches, Width and Height
# * Ppi(Pixels Per Inch) can be introduced and Inches, Width, and height can be dropped
# 

# In[89]:


# Introducing a new Column 'Ppi' by calculating value from inches, width and height and dropping inches, width and height columns

#calculating ppi value using the formula: ppi = diagonal(in inches) / diagonal(in pixels) and creating a new column to store it
df['Ppi'] = ((df['Width'] ** 2) + (df['Height'] ** 2)) ** 0.5 / df['Inches']

# dropping necessary columns
df.drop(columns = ['Inches', 'Width', 'Height'], inplace = True)

# viewing the dataframe
df.sample(10)


# In[90]:


plt.scatter(x = df['Ppi'], y = df['Price'])
plt.xlabel('Ppi')
plt.ylabel('Price')
plt.show()


# In[91]:


# finding the correlation coefficient between price and Ppi from correlation matrix

df.corr()['Price']['Ppi']


# In[92]:


# Displaying the correlation matrix

df.corr()


# In[93]:


# displaying the heatmap for the above correlation matrix for visualization

sns.heatmap(df.corr())
plt.show()


# In[94]:


# plotting the distribution of log(price)

sns.histplot(np.log(df['Price']), kde = True)
plt.show()


# In[95]:


# storing the values of the target column price after log transformation
y = np.log(df['Price'])

# viewing the current data
y


# In[96]:


# storing the values of the rest of the indepedant variables
X = df.drop(columns = 'Price')

# viewing the data
X


# # Splitting the data into test set and train set

# ## Neccessary Libraries and Modules required :
# 
# * train_test_split module in sklearn.model_selection library

# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


# splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15 )


# In[99]:


# viewing the X_train set

X_train


# In[100]:


# Viewing the y_train set

y_train


# In[101]:


# viewing the X_test set

X_test


# In[102]:


# viewing the y_test

y_test


# # Building ML Model

# ## Neccessary Libraries required:
# 
# 
# #### For Column Transformation of categorical variables:
# 
# * ColumnTransformer module in sklearn.compose library
# * OneHotEncoder module in sklearn.preprocessing library
# 
# #### For Pipeline object
# 
# * Pipeline module in sklearn.pipeline library
# 
# 
# #### For Linear Regression ML Model
# 
# * LinearRegression module in sklearn.linear_model library
# * r2_score module module in sklearn.metrics library
# * mean_absolute_error module in sklearn.metrics library

# In[103]:


# importing neccessary modules from neccessary libraries

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# ## Linear Regression Model

# In[104]:


# building the pipeline structure for linear regression model

# step-1: applying column transformtion on the nominal categorical variables
col_transformer = ColumnTransformer( transformers = [
    ('nom_trans',OneHotEncoder(sparse = False, drop = 'first'), [0, 1, 6, 9, 10] )
], remainder = 'passthrough')

# step-2: creating a linear regression model after column transformations
regressor = LinearRegression()

# creating a pipe object for pipelining the above steps
pipe = Pipeline([
    ('Step1_col_trans', col_transformer),
    ('step2_regressor', regressor)
])

# passing the training data sets to the pipe object 
pipe.fit(X_train, y_train)

# predicting the target column y by passing the x test set
y_pred = pipe.predict(X_test)


# ### Model Performance

# In[105]:


print("R2 Score :", r2_score(y_test, y_pred))
print("Mean Absolute Error(MAE):", mean_absolute_error(y_test, y_pred))


# # Model Deployment

# ### Required Libraries:
# 
# * pickle

# In[106]:


# importing neccessary library

import pickle


# In[107]:


# Exporting the preprocessed and cleaned dataframe

df.to_csv('df.csv', index = False)


# In[108]:


# deploying the model

pickle.dump(pipe, open('model.pkl', 'wb'))

