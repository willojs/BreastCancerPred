library(corrplot)
library(caret)
library(caTools)
library(ggplot2)
library(e1071)
library(dplyr)
library(nnet)
library(RCurl)
library(class)
library(readxl)
library(pROC)


#Read data in
data <- read_excel("~/BreastCancerPred/wisc_bc_data.xlsx")

#Converting diagnosis to factor
data$diagnosis <- as.factor(data$diagnosis)

#Making id null
data$id <- NULL

summary(data)

#Checking for data imbalance
prop.table(table(data$diagnosis))

#Splitting the data
set.seed(10)
split = sample.split(Y = data$diagnosis, SplitRatio = 0.8)
training.set = subset(data, split == TRUE)
test.set = subset(data, split == FALSE)


#Applying machine learning models
fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

##Naive Bayes
#Building the model
start.time.nb <- Sys.time()
model_nb <- train(diagnosis~.,
                  training.set,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)

## predicting for test data
pred_nb <- predict(model_nb, test.set)
cm_nb <- confusionMatrix(pred_nb, test.set$diagnosis, positive = "M")
cm_nb
end.time.nb <- Sys.time()
time.taken.nb <- end.time.nb - start.time.nb
time.taken.nb

##KNN
#Building the model
start.time.knn <- Sys.time()
model_knn <- train(diagnosis~.,
                   training.set,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)

##predicting for test data
pred_knn <- predict(model_knn, test.set)
cm_knn <- confusionMatrix(pred_knn, test.set$diagnosis, positive = "M")
cm_knn
end.time.knn <- Sys.time()
time.taken.knn <- end.time.knn - start.time.knn
time.taken.knn

##Random Forest
#Building the model
start.time.rf <- Sys.time()
model_rf <- train(diagnosis~.,
                  training.set,
                  method="ranger",
                  metric="ROC",
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)

##predicting for test data
pred_rf <- predict(model_rf, test.set)
cm_rf <- confusionMatrix(pred_rf, test.set$diagnosis, positive = "M")
cm_rf
end.time.rf <- Sys.time()
time.taken.rf <- end.time.rf - start.time.rf
time.taken.rf

#Model Comparison
model_list <- list(RF=model_rf, KNN=model_knn, NB=model_nb)
resamples <- resamples(model_list)

#Correlation between models
model_cor <- modelCor(resamples)


corrplot(model_cor, method = "number")


par(mfrow = c(4, 2))
# Random Forest ROC
roc_rf <- plot(roc(response = test.set$diagnosis,  
                   predictor = as.numeric(as.factor(pred_rf))),
               col = '#9C27B0', main = 'Random Forest')
auc(roc_rf)


# Naive Bayes ROC
roc_nb <- plot(roc(response = test.set$diagnosis,  
                   predictor = as.numeric(pred_nb)),
               col = '#FF9800', main = 'Naive Bayes')
auc(roc_nb)

# K-NN ROC
roc_knn <- plot(roc(response = test.set$diagnosis,  
                    predictor = as.numeric(pred_knn)),
                col = '#795548', main = 'K-NN')
auc(roc_knn)

bwplot(resamples, metric="ROC")

cm_list <- list(RF=cm_rf, KNN=cm_knn, NB=cm_nb)
cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results

cm_results_max <- apply(cm_list_results, 1, which.is.max)

report <- data.frame(metric=names(cm_results_max), 
                     best_model=colnames(cm_list_results)[cm_results_max],
                     value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                  names(cm_results_max), 
                                  cm_results_max))
rownames(report) <- NULL
report
