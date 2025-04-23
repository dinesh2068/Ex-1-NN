import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer as sim  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 

#Read the dataset from drive
dataset = pd.read_csv("Ex-1-NN\Churn_Modelling.csv")
print(dataset.head())
print(dataset.tail())

# Finding Missing Values
print("Missing Values: \n ",dataset.isnull().sum())

# hadling the missing dataset's using the mean in number column
impute_txt = sim(strategy='mean')
text_col = dataset.select_dtypes(include=[np.number]).columns
dataset[text_col] = impute_txt.fit_transform(dataset[text_col])

#Check for Duplicates
print("Duplicate values:\n ")
print(dataset.duplicated())

# encoding the categorical data (Text) into number (since ml model can't understand the missing data)
label = {}
for i in text_col:
    le = LabelEncoder()
    dataset[i] = le.fit_transform(dataset[i])
    label[i] = le

# Normalizing the data to perform algorithm's better
scale = StandardScaler()
dataset[text_col] = scale.fit_transform(dataset[text_col])
print("Normalized data for column(s):", text_col)
print(dataset[text_col])

# splitting the datasets to train the model     
x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Print the training data and testing data
print("Training data")
print(x_train)
print(y_train)

print("Testing data")
print(x_test)
print(y_test)
print("Length of X_test: ", len(x_test))