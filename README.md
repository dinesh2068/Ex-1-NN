`<H3>`ENTER YOUR NAME`</H3>`
`<H3>`ENTER YOUR REGISTER NO.`</H3>`
`<H3>`EX. NO.1`</H3>`
`<H3>`DATE`</H3>`
`<H1 ALIGN =CENTER>` Introduction to Kaggle and Data preprocessing`</H1>`

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

## ALGORITHM:

STEP 1:Importing the libraries`<BR>`
STEP 2:Importing the dataset`<BR>`
STEP 3:Taking care of missing data`<BR>`
STEP 4:Encoding categorical data`<BR>`
STEP 5:Normalizing the data`<BR>`
STEP 6:Splitting the data into test and train`<BR>`

## PROGRAM:

```

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

```

## OUTPUT:

### df.head(): 

![1745419785249](image/README/1745419785249.png)

### df.tail():

![1745419865415](image/README/1745419865415.png)

### Missing Values:

![1745419936039](image/README/1745419936039.png)

### Duplicate values:

![1745419982890](image/README/1745419982890.png)

### Normizing Data:

![1745420864002](image/README/1745420864002.png)

### Training data:

![1745421005769](image/README/1745421005769.png)

### Testing data:

![1745421042164](image/README/1745421042164.png)

## RESULT:

Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
