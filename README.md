# Analyze-Insurance-Payout
We will analyze how multiple factors (E.g. Age, Sex and Smoking status) affect the charges of the insurance company for clients

Dowload the dataset from 

https://www.kaggle.com/mirichoi0218/insurance

Before running the R file, remember to install all of these packages 

```
library(psych)
library(dplyr)
library(car)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(caret)
library(keras)
library(tfdatasets)
library(e1071)
library(pROC)
```

Run the R file to receive results

**Overview of analyzation steps**

* Give a brief background of the data selected
* Using Multiple Linear Regression to analyse the relationship between independent variables and “charges”
* Using Support Vector Machines to predict "log(charges)"
* Using Logit Regression Model and Decision tree to classify the new variable "highcharges"
