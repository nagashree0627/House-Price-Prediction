#!/usr/bin/env python
# coding: utf-8

# **IMPORT LIBRARIES**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# **TASK-1 : DATA LOADING AND EXPLORATION**

# In[5]:


df=pd.read_csv(r"C:\Users\ADMIN\Downloads\Housing.csv")


# In[6]:


df.head(10)


# In[7]:


print("Rows:",df.shape[0])
print("Columns:",df.shape[1])


# In[12]:


df.columns


# **Target and Features**

# In[11]:


target="price"

features=df.drop("price",axis=1)
print("Target Column:",target)
print("Feature Columns:")
print(features.columns)


# In[14]:


df.isnull().sum()


# In[21]:


df.info()


# In[13]:


df.describe()


# **TASK-2 : DATA CLEANING**

# In[16]:


df=df.drop_duplicates()


# In[17]:


df.fillna(df.median(numeric_only=True),inplace=True)


# In[19]:


for col in df.select_dtypes(include=["object","string"]):
    df[col]=df[col].fillna(df[col].mode()[0])


# In[20]:


df=pd.get_dummies(df,drop_first=True)


# **TASK-3 : MODEL BUILDING**

# In[22]:


X=df.drop("price",axis=1)
y=df["price"]


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# *Linear Regression*

# In[26]:


model1=LinearRegression()
model1.fit(X_train,y_train)
y_pred_model1=model1.predict(X_test)


# In[28]:


mae_model1=mean_absolute_error(y_test,y_pred_model1)
rmse_model1=np.sqrt(mean_absolute_error(y_test,y_pred_model1))
r2_model1=r2_score(y_test,y_pred_model1)
print("Linear Regression")
print("MAE: ",mae_model1)
print("RMSE: ",rmse_model1)
print("R2: ",r2_model1)


# *Random Forest*

# In[29]:


model2=RandomForestRegressor(n_estimators=100,random_state=42)
model2.fit(X_train,y_train)
y_pred_model2=model2.predict(X_test)


# In[30]:


mae_model2=mean_absolute_error(y_test,y_pred_model2)
rmse_model2=np.sqrt(mean_absolute_error(y_test,y_pred_model2))
r2_model2=r2_score(y_test,y_pred_model2)
print("Random Forest")
print("MAE: ",mae_model2)
print("RMSE: ",rmse_model2)
print("R2: ",r2_model2)


# *MODEL COMPARISON*

# In[33]:


comparison=pd.DataFrame({
    "Model":["Linear Regression","Random Forest"],
    "MAE":[mae_model1,mae_model2],
    "RMSE":[rmse_model1,rmse_model2],
    "R2 Score":[r2_model1,r2_model2]
})
comparison


# **EVALUATION**

# **VISUALIZATIONS**

# *HISTOGRAM*

# In[49]:


plt.figure(figsize=(8,5))
sns.histplot(df['price'],bins=30,kde=True)
plt.title("House Price Distribution")
plt.show()


# *CORRELATION HEATMAP*

# In[48]:


plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# *ACTUAL Vs PREDICTED*

# In[50]:


plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred_model2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Vs Predicted")
plt.show()


# **Feature Importance**

# In[38]:


imp=pd.DataFrame({
    'Feature': X.columns,
    'Importance':model2.feature_importances_
})
imp=imp.sort_values(by='Importance',ascending=False)
imp.head(10)


# In[40]:


plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=imp.head(10))
plt.title("Top 10 Important Features")
plt.show()


# **TASK-5 INSIGHTS AND SUMMARY**

# 1. Which features influence house price the most?
# 
# Area of the house
# Number of bathrooms
# Air conditioning availability
# Preferred area
# Number of stories
# 
# 2. How accurate was the model?
# 
# Random Forest achieved a higher R² score than Linear Regression.
# It predicted house prices with good accuracy and lower prediction error.
# 
# 3. What surprised you?
# 
# Area had a much stronger impact on price than the number of bedrooms.
# Houses in preferred locations showed significantly higher prices.
# 
# 4. Recommendation
# 
# Real estate companies should focus on larger houses in preferred locations because these features contribute most to higher property values.

# In[ ]:




