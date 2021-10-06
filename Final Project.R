#Load Packages

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

############################################################################## Importing data

#Import Dataset
df <- read.csv(file.choose(), header = T)

#IdealBMI variable
df$idealbmi <- NA
df$idealbmi[df$bmi >= 18.5 & df$bmi <= 24.9] <- 1
df$idealbmi[df$bmi < 18.5 |df$bmi > 24.9] <- 0


#Split data to 50:50 training and test data
set.seed(123)
index <- as.numeric(row.names(df))
n <- nrow(df)
testindex <- sample(index, n*0.5)
#train.data
train.data <- df[testindex,]
#test.data
test.data <- df[-testindex,]


################################################################# Multiple Linear Regression


######################### Linear Regression 1: All independent variables included #########################
# Model Training
lm1 <- lm(log(charges) ~ log(age) + sex + idealbmi + children + smoker + region,train.data)
summary(lm1)

# Model Prediction
pred1 <- lm1 %>% predict(test.data)

# Model Analysis
RMSE(pred1, log(test.data$charges))
R2(pred1, log(test.data$charges))

# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred1, col = "red", pch = 4)

# Assumption: Multicollineary
car::vif(lm1)

# Assumption: Normality
qqnorm(pred1-log(test.data$charges))
qqline(pred1-log(test.data$charges))

# Assumption: Homoskedasticity
plot(pred1, log(test.data$charges) - pred1)



###################### Linear Regression 2: Variable Region excluded #########################
# Model Training
lm2 <- lm(log(charges) ~ log(age) + sex + idealbmi + children + smoker,train.data)
summary(lm2)

# Model Prediction
pred2 <- lm2 %>% predict(test.data)

# Model Analysis
RMSE(pred2, log(test.data$charges))
R2(pred2, log(test.data$charges))

# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred2, col = "red", pch = 4)

# Assumption: Multicollineary
car::vif(lm2)

# Assumption: Normality
qqnorm(pred2-log(test.data$charges))
qqline(pred2-log(test.data$charges))

# Assumption: Homoskedasticity
plot(pred2, log(test.data$charges) - pred2)



###################### Linear Regression 3: Variable Sex excluded  #########################
# Model Training
lm3 <- lm(log(charges) ~ log(age) + idealbmi + children + smoker + region,train.data)
summary(lm3)

# Model Prediction
pred3 <- lm3 %>% predict(test.data)

# Model Analysis
RMSE(pred3, log(test.data$charges))
R2(pred3, log(test.data$charges))

# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred3, col = "red", pch = 4)

# Assumption: Multicollineary
car::vif(lm3)

# Assumption: Normality
qqnorm(pred3-log(test.data$charges))
qqline(pred3-log(test.data$charges))

# Assumption: Homoskedasticity
plot(pred3, log(test.data$charges) - pred3)


###################### Linear Regression 4: Variable Sex and Region excluded  #########################
# Model Training
lm4 <- lm(log(charges) ~ log(age) + idealbmi + children + smoker,train.data)
summary(lm4)

# Model Prediction
pred4 <- lm4 %>% predict(test.data)

# Model Analysis
RMSE(pred4, log(test.data$charges))
R2(pred4, log(test.data$charges))

# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred4, col = "red", pch = 4)

# Assumption: Multicollineary
car::vif(lm4)

# Assumption: Normality
qqnorm(pred4-log(test.data$charges))
qqline(pred4-log(test.data$charges))

# Assumption: Homoskedasticity
plot(pred4, log(test.data$charges) - pred4)


################################################################# Support Vector Machine
# Model Training
svmfit_1 = svm(log(charges) ~ log(age) + idealbmi + children + smoker, data = train.data)
print(svmfit_1)


# Model Prediction
pred_svm1 <- predict(svmfit_1, test.data)


# Model Analysis
RMSE(pred_svm1, log(test.data$charges))
R2(pred_svm1, log(test.data$charges))


# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred_svm1, col = "red", pch = 4)


################################################################# Regression Tree

# Model Training
rtree1 <- rpart(log(charges) ~ log(age) + children + smoker + as.factor(idealbmi) + region + sex, data = train.data)

# Model Visualisation
prp(rtree1)
rpart.plot(rtree1, roundint = FALSE , digits = 4)


# Model Prediction
pred_rt1 <- predict(rtree1, test.data)


# Model Analysis
RMSE(pred_rt1, log(test.data$charges))
R2(pred_rt1, log(test.data$charges))


# Prediction Visualisation
plot(log(test.data$age), log(test.data$charges), col = "blue")
points(log(test.data$age), pred_rt1, col = "red")




###############################################################################################################


#New Classification Variable
df$highcharge <- NA
df$highcharge[df$charges > 15000] <- 1
df$highcharge[df$charges <= 15000] <- 0
df$highcharge <- as.factor(df$highcharge)



#Split data to 70:30 training and test data
set.seed(123)
index <- as.numeric(row.names(df))
n <- nrow(df)
testindex <- sample(index, round(n*0.7))
#train.data
train.data <- df[testindex,]
#test.data
test.data <- df[-testindex,]



################################################################# Logit Model
# Model Training
fit_log1 <- glm(highcharge ~ age + children + smoker + idealbmi, family = binomial(link="logit"), train.data)
summary(fit_log1)

# Model Probability Prediction
pred_log1 <- predict(fit_log1, test.data, type = "response")

#Model Class Prediction
predclass_log1 <- ifelse(pred_log1 > 0.5, 1, 0)

# Model Analysis
confusionMatrix(table(predicted = predclass_log1, actual = test.data$highcharge))


################################################################# Decision Tree Model
# Model Training
fit_dt1 <- rpart(highcharge ~ age + children + smoker + idealbmi + sex + region, data=train.data,  method="class", parm=list(split="gini"))

# Model Class Prediction
pred_dt1 <- predict(fit_dt1, test.data, type = "class")

# Model Analysis
confusionMatrix(table(predicted = predclass_dt1, actual = test.data$highcharge))

# Model Visualisation
prp(fit_dt1)
rpart.plot(fit_dt1, roundint = FALSE , digits = 4)

################################################################# ROC, AUC Comparison


par(pty="s")
roc(test.data$highcharge, pred_log1, plot=TRUE, legacy.axes=TRUE, print.auc=TRUE, col="red", main = "ROC for Logit Regression Model")
roc(test.data$highcharge, as.numeric(pred_dt1), plot=TRUE, legacy.axes=TRUE, print.auc=TRUE, col="blue", main = "ROC for Decision Tree Model")

