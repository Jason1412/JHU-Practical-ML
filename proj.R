#
# Practical Machine Learning
#
# Project

data = read.csv('pml-training.csv')
pred = read.csv('pml-testing.csv')

set.seed(1)
inTrain = createDataPartition(data$classe, p = 0.8, list = FALSE)


training = data[inTrain,]
testing = data[-inTrain,]


# Check the summary of the data
str(training)

summary(training)



# count NAs of each column
count_NA = apply(training, 2, function(x) sum(is.na(x)))
plot(count_NA)
summary(count_NA)

# find the column which has too many NAs
ind = count_NA > 1000

# delete the columns with too many NAs
training1 = training[,!ind]
testing1 = testing[,!ind]
pred1 = pred[,!ind]
# delete the following columns:
# X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp

library(tidyr)
library(dplyr)
training2 = select(training1, -(X:new_window))
testing2 = select(testing1, -(X:new_window))
pred2 = select(pred1, -(X:new_window))

# find the columns with the name containing "amplitude"
# delete the columns with yaw_forearm, yaw_dumbbell, yaw_belt
ind_amp = sapply(names(training2), function(x) grepl("amplitude",x) | grepl("yaw_forearm",x) | grepl("yaw_dumbbell", x) | grepl("yaw_belt", x))


training3 = training2[,!ind_amp]
testing3 = testing2[,!ind_amp]
pred3 = pred2[,!ind_amp]

# find the factors of all columns
# index of factor
ind_fac = sapply(training3, function(x) is.factor(x))

ind_fac[length(ind_fac)] = FALSE # neglect the last column in factor

sum(ind_fac)

# convert the factor to numeric
tmp = sapply(training3[,ind_fac], function(x) as.numeric(levels(x))[x])
# Observation -------------------------------------------------------------
# We find NAs after the forcing the column to be factors 
# -------------------------------------------------------------------------

# if deleting all the factor columns
training4 = training3[,!ind_fac]
testing4 = testing3[,!ind_fac]
pred4 = pred3[,!ind_fac]
pred4 = pred4[,-ncol(pred4)]




# Use hierarchical cluster analysis to see the grouping
hc_data = dist(as.matrix(training4))
hc = hclust(hc_data)
plot(hc)


# Decision tree to fit the value =====================================================
library(rpart)

library(rattle)
ind_spl = sample(1:nrow(training4), nrow(training4))
fit = rpart(classe~., method = 'class', data = training4)

# printcp(fit)
# plotcp(fit)
# summary(fit)

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree")

fancyRpartPlot(fit)

# prediction on the testing data
fit_pred = predict(fit, newdata = testing4, type = 'class')
confusionMatrix(fit_pred, testing4$classe)


# random forest =======================================================================
library(caret)
rf_fit = train(classe ~., method = "rf", data = training4)

library(randomForest)
rf_fit0 = randomForest(classe ~., data = training4, ntree = 100, mtry = 7)
varImp(rf_fit0)
varImpPlot(rf_fit0,type=2)

fit0_pred = predict(rf_fit0, newdata = testing4, type = 'class')
confusionMatrix(fit0_pred, testing4$classe)
fit0_pred



# check on the pred data set =========================================================
pred_data = predict(rf_fit0, newdata = pred, type = 'class')
pred_data

