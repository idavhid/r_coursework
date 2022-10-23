#Link to document: https://docs.google.com/document/d/1-cBHDWnf473ALee5he5QzqsbR8CHlelJDUeVduvnVOE/edit#

# To Do
# -library(readr)
# -clustering hierarchical k-means with Hartigan's Rule
# -caret
# -decision tree
# -SVM (lib kernlab)
# -neural network
# - logistic regression (plus lasso, plus bootstrapping)
# - shapley shapr: Explaining individual machine learning predictions with Shapley values
# better to use shapper 
# in the end use ensemble https://github.com/RichardOnData/YouTube-Scripts/blob/master/R%20Tutorial%20(ML)%20-%20caret.Rmd
# kfold with stratified, use oversampling? 
# 
# 
#split = sample.split(df$deceased, SplitRatio = 0.70)
#df_train = subset(df, split == TRUE)
#df_test = subset(df, split == FALSE)

#df_train <- 

#data normalization
#https://datascience.stackexchange.com/questions/13971/standardization-normalization-test-data-in-r

# misClasificError <- mean(dummy_predictions != df_train$deceased)
# print(paste('Accuracy',1-misClasificError))


# misClasificError <- mean(fitted.results != df_test$deceased)
# print(paste('Accuracy',1-misClasificError))


##FRIEDHOF

# # check for multicollinearity and verify if dummy encoding has worked
# #One-hot encoding converts it into n variables, while dummy encoding converts it into n-1 variables
# df_num <- df_train %>% select(where(is.numeric))
# 
# #
# df_num %>%
#   cor(use="pairwise.complete.obs", method="spearman") %>%
#   ggcorrplot(show.diag = F, type="full", lab=TRUE, lab_size=2, outline.color = "white",
#              ggtheme = ggplot2::theme_gray,
#              colors = c("#6D9EC1", "white", "#E46726"))
# 
# 
# 
# #vif werte
# 
# library("car")
# 
# vif(lr_model)






# MSc Data Science
# Module: R for Data Science (DSM110)
# 
# Midterm Coursework: Data Cleaning 
# Session: April 2022
# 
# Student Number: 200199830
# Created: 15 July 2022
# Last modified: 02 September 2022


# ******************************************************************************
#                     1.1 define functions for modular code
# ******************************************************************************

#making sure that the environment is empty
rm(list=ls())


#define function that verifies if required packages are already installed; if they are not proceed with installation,
#otherwise load package
#define function
package_checker <- function(pack_list){
  #select CRAN mirror for downloading of packages as instructed by course instructors
  options(repos="https://cran.ma.imperial.ac.uk")
  #suppresses possible messages about availability of binary or source file
  #as instructed by course instructors
  options(install.packages.check.source = "no")
  
  #iterate over requested packages
  for (pack in pack_list) {
    #if requested package is not part of installed package then install package
    if (pack %in% rownames(installed.packages())== FALSE)
    {print(paste("Can not find package ",pack,", initialize install process...",sep = ""))
      install.packages(pack)
      library(package = pack, character.only = TRUE)
    } 
    
    #if requested package is part of installed package then load package
    else{
      print(paste("loading ", pack,"...",sep = ""))
      
      library(package = pack, character.only = TRUE)
    }
  }
}




# ******************************************************************************
#                     1.2 install / library - initialize packages
# ******************************************************************************

# 1) setting seed for any random function for reproducibility
set.seed(123)

# 2) set path to that folder where R script is located, then access data folder 
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
wd_path <- "./data/"
setwd(wd_path)

# 3) installing (if not present) and loading packages

#Required packages: "useful","tidyverse", "patchwork", "mice", "tibble", "ggplot2"

package_checker(c("caret",     #streamline machine learning
                  "readr",     #read in csv files
                  "tidyverse", #data wrangling
                  "tibble",    #stricter datatable format
                  "ggplot2",   #visualization
                  "kernlab",   #support vector machines
                  "randomForest", #random forest
                  "glmnet",    #lasso logistic regression
                  "pROC" ,     #function for AUC ROC, precision, recall, confusion matrix
                  "gbm",       #gradient boosting machines
                  "neuralnet", #neural networks
                  "class",     #KNN
                  "MLmetrics", #individual printing of f1 score, precision, recall
                  "caretEnsemble" #ensembling of models
))

#if the custom function "package_checker" does not work (although this was 
#tested on different computers), please uncomment the following and run it:
# install.packages("tidyverse")
# install.packages("readr")
# install.packages("caret")
# install.packages("tibble")
# install.packages("ggplot2")
# install.packages("kernlab")
# install.packages("randomForest")
# install.packages("glmnet")
# install.packages("pROC")
# install.packages("gbm")
# install.packages("neuralnet")
# install.packages("class")
# install.packages("MLmetrics")
# install.packages("caretEnsemble")

# library("tidyverse")
# library("readr")
# library("caret")
# library("tibble")
# library("ggplot2")
# library("kernlab")
# library("randomForest")
# library("glmnet")
# library("pROC")
# library("gbm")
# library("neuralnet")
# library("class")
# library("MLmetrics")
# library("caretEnsemble")

# ******************************************************************************
#                     2.1 Loading data sets.
# ******************************************************************************

# Using trycatch to load datasets
#if datasets cannot be loaded print error message and current working directory
#to help with debugging
tryCatch(                       
  
  expr = {                      
    #read in cleaned and processed dataset, change to tibble datatype
    df <- read_csv("2022-07-15_cleaned_dataset.csv") %>% as_tibble()
    
    message("Cleaned dataset was successfully loaded.")
  },
  
  error = function(e){          
    message("The cleaned dataset could not be loaded. Please make sure that current working directory is within the data folder where the files are located.")
    print(getwd())
  })

#if manual reading in of dataframes outside of trycatch is required
# df <- read_csv("2022-07-15_cleaned_dataset.csv") %>% as_tibble()


# ******************************************************************************
#                     2.2 Feature selection
# ******************************************************************************

#remove weight and height as derived feature BMI is included
df <- subset(df, select = c(-height,-weight ))

#on the 16th of June switch of standard treatment with dexamethasone being
#given to all patients with more severe forms of COVID
df$dexa_standard_period <- (df$date_admission_hospital > "2020-06-16") %>% as.integer() 

#after encoding dexamethasone treatment, date of hospital admission is not useful anymore
#additionally referral from physician and patient ID have no further predictive value (as explained by clinician)
df <- subset(df, select = c(-PatID,-date_admission_hospital, -referral))

#organ replacement with ECMO (artifical lung) and dialysis(artificial kidney), 
#mechanical ventilation (intubation) all only occur after admission 
#to avoid information leak from the future removing these variables
#leave "intubated_referral" in dataset as this is known at admission
df <- subset(df, select = c(-ecmo_treatment,-dialysis_treatment, -intubation
                            ))

#define all columns that contain info about preconditions
precondition_cols <-  c("hypertension",
   "cardiovascular_disease",
   "asthma",
   "copd",
   "diabetes",
   "diabetes_complication",
   "hypercholesterolemia",        
   "kidney_disease"       ,        "dialysis"             ,        "liver_disease"          ,     
   "neurological_disease"  ,       "stroke"                )

#using row sum function to create new feature that counts up all preconditions
df$number_precondition <- rowSums(df[,precondition_cols] == 1)

#identify near zero variance predictors and assess if they can be removed 
low_varinace_cols <-  nearZeroVar(df,freqCut = 90/10, uniqueCut = 10)

#print out columns that have very low variance
print(df %>% select(low_varinace_cols) %>% colnames())

#removing low variance features (except DNR ("do not resuscitate") and COPD (chronic lung disease) 
#as these are actually informative albeit only a subset of patients have )
df <- df %>% select(-c("asthma", "diabetes_complication", "dialysis", "liver_disease",
                 "stroke", "immunosuppression", "smoker", "neurological_disease"))

#encode all factor columns
factor_cols <-  c("hypertension",
                  "cardiovascular_disease",
                  "copd",
                  "diabetes",
                  "hypercholesterolemia",        
                  "kidney_disease",  
                  "intubated_referral",
                  "deceased",   
                  "dnr_dni",
                  "female",
                  "dexa_treatment",
                  "dexa_standard_period"
)

#using vectorized function to change the column type
df[factor_cols] <- lapply(df[factor_cols], factor)


# ******************************************************************************
#                     3 Model building
#                     3.1 Train/Test Split, dummy encoding and data normalization
# ******************************************************************************


#get random stratified sample (stratified by outcome)
#get row numbers for training data
set.seed(123)
train_row_number <- df$deceased %>% createDataPartition(p = 0.7, list = FALSE)

#seperate data based on row numbers into training and testing set
df_train  <- df[train_row_number, ]
df_test <- df[-train_row_number, ]


#use training data to calculate mean and standard deviation
normalizing_param <- preProcess(df_train, method = c("center", "scale"))

#use mean and standard deviation of training data to normalize training and test data
df_train <- predict(normalizing_param, df_train)
df_test <- predict(normalizing_param, df_test)

#dummy encoding of training dataset
dummy_encoding_train <- dummyVars(deceased ~ ., data = df_train, 
                                  fullRank=T) #no linear dependencies are introduced 
df_train_no_target <- as.data.frame(predict(dummy_encoding_train, newdata = df_train))
df_train <- cbind(df_train_no_target, df_train$deceased)
df_train <- df_train %>% rename(deceased="df_train$deceased")

#dummy encoding of test dataset
dummy_encoding_test <- dummyVars(deceased ~ ., data = df_test, fullRank=T)
df_test_no_target <- as.data.frame(predict(dummy_encoding_test, newdata = df_test))
df_test <- cbind(df_test_no_target, df_test$deceased)
df_test <- df_test %>% rename(deceased="df_test$deceased")


#after dummy endoding standardize name of columns to avoid characters such as 
#"()" that cause problems in model functions
names(df_train) <- make.names(names(df_train))
names(df_test) <- make.names(names(df_test))

#caret expects target column as factors and not numeric
df_train$deceased <- as.factor(if_else(df_train$deceased == 1, "deceased", "survival"))
df_test$deceased <- as.factor(if_else(df_test$deceased == 1, "deceased", "survival"))


# ******************************************************************************
#                     3.2 Define predictor variable sets
# ******************************************************************************

#use colname function to retrieve all column names, using vector indexing to only
#get clinical variables (that come first in order)
clinical_cols <- colnames(df_train)[0:12] 

#must append three clinical variables that are located at different positions in 
#original vector
clinical_cols <- append(clinical_cols, c("bmi", "dexa_standard_period.1", "number_precondition"))

#verify that all clinical variables were retrieved
print(clinical_cols)

#select all columns as laboratory variables except clinical variables and deceased 
#which is the outcome (target)
laboratory_cols <- df_train %>% select(-one_of(append(clinical_cols, "deceased"))) %>% colnames()

#verify that all laboratory variables are present
print(laboratory_cols)

#encode all numerical columns - needed for feature plot
numeric_cols <- df_train %>% select(append(laboratory_cols, c("age","bmi", "number_meds", "number_precondition"))) %>% colnames()

#display all numerical features in boxplots stratified by outcome
featurePlot(x = df_train[,numeric_cols],
            y = df_train$deceased,
            plot = "box")

# ******************************************************************************
#                     3.3 Define functions for model building and evaluation
# ******************************************************************************


#define parameters that are being used during training
train_control_cv <- trainControl(
  method = 'cv', number = 5,       #as resampling method, perform k-fold cross validation with 5 folds
  savePredictions = 'final',       #predictions are saved
  classProbs = T,                  #class probabilities will be returned
  summaryFunction=twoClassSummary, #defines summary
) 

#define modified training parameters where minority class oversampling is used
train_control_cv_upsampling <- trainControl(
  method = 'cv', number = 5, savePredictions = 'final', classProbs = T, summaryFunction=twoClassSummary, 
  #oversampling - minority class is sampled multiple times but train/validation split of cross-validation
  #is respected, so no information leak during training occurs
  sampling = "up" 
) 



model_comparison <- function(name_model, name_model_output){
  
  #initialize variables that will be used for summary dataframe
  current_predictors_range <- NULL
  accuracy_range <- NULL
  precision_range <- NULL
  recall_range <- NULL
  f1_score_range <- NULL
  auc_range <- NULL
  
  #except in last model training approach, no oversampling is used
  training_parameters = train_control_cv
  
  #for loop that performs model training and evaluation on separate selection of predictors
  for (i in c(1:4)){
    #only use clinical predictors 
    if (i == 1){
      current_predictors <- "clinical"
      training_data <- df_train %>% select(clinical_cols, deceased)
      testing_data <- df_test %>% select(clinical_cols, deceased)
    }
    #only use laboratory predictors 
    else if (i == 2){
      current_predictors <- "laboratory"
      training_data <- df_train %>% select(laboratory_cols, deceased)
      testing_data <- df_test %>% select(laboratory_cols, deceased)
    }
    #use full dataset for predictions 
    else if (i == 3){
      current_predictors <- "clinical & laboratory"
      training_data <- df_train 
      testing_data <- df_test 
    }
    #use full dataset and oversampling of minority class
    else{
      current_predictors <- "clinical & laboratory, oversampling"
      training_parameters = train_control_cv_upsampling
    }
    
    #train model with cross validation
    current_model<- train(form=deceased~., 
                          data=training_data, 
                          metric="ROC",                  #metric for optimization  
                          trControl=training_parameters, #use cross validation
                          method=name_model) #specify model name
    
    #calculate predictions
    model_prob <- predict(object=current_model,newdata=testing_data,type='prob')$survival
    
    #use calculated probabilities to assign class label based on 0.5 threshold
    #transform to factor datatype
    model_predict_labels <- as.factor(ifelse(model_prob > 0.5,"survival","deceased"))
    
    #call evaluation function to calculate and store model performance in variable
    performance_metrics_model <- model_eval(predict_label=model_predict_labels, true_labels=testing_data$deceased,
                                            predictions=model_prob, title_plot = paste(name_model_output," (",current_predictors,")",sep = "")
                                            ,roc=TRUE)
    
    #unpacking model performance metrics that are stored in a list
    #building up numeric vectors that can subsequently be used to build a summary
    #dataframe
    current_predictors_range[i] <- current_predictors
    accuracy_range[i] <- performance_metrics_model$accuracy
    precision_range[i] <- performance_metrics_model$precision
    recall_range[i] <- performance_metrics_model$recall
    f1_score_range[i] <- performance_metrics_model$f1
    auc_range[i] <- performance_metrics_model$auc
    
    #print variable importance if random forest model is selected
    if (name_model =="rf"){
      var_importance <- varImp(current_model,scale=TRUE)
      var_plot <- plot(var_importance,
      main=paste("Feature importance with predictors: ",
                 current_predictors,sep = ""),
      xlab="Scaled feature importance")
      print(var_plot)
    }
  }
  #build summary dataframe 
  summary_dataframe <- data.frame(current_predictors_range, accuracy_range, precision_range,
                                  recall_range, f1_score_range, auc_range)
  #change column names to better readable names
  colnames(summary_dataframe) <- c("Predictor variables", "Accuracy", "Precision",
                                   "Recall", "F1 score", "AUC")
  
  #write model performance to csv
  write.csv(summary_dataframe, paste(name_model_output,"_metrics.csv",sep = ""), row.names = F)
  
  return(summary_dataframe)
}


#define function to evaluate models with confusion matrix and if specified also
#calculate area under curve
model_eval <- function(predict_label, true_labels, predictions, print_out =FALSE, title_plot="", roc=FALSE){
  
  #transform into factor variable (required for calculation of auc) and making sure
  #that level of labels is consistent between ground truth and prediction
  true_labels <- as.factor(true_labels)
  predict_label <- factor(predict_label, levels=levels(true_labels))
  
  #calculate confusion matrix
  confusion_matrix <- table(predict_label,true_labels)
  
  #calculate accuracy, precision, recall, F1-score, auc
  current_accuracy <- round(Accuracy(y_true= true_labels, y_pred = predict_label),2)
  current_precision <- round(Precision(y_true= true_labels, y_pred = predict_label, positive="deceased"),2)
  current_recall <- round(Recall(y_true= true_labels, y_pred = predict_label, positive="deceased"),2)
  if (predictions == FALSE){
    current_auc = NA
  }
  else {
  current_auc <- round(auc(true_labels, predictions),2)
  }
  #calculate F1 score based on standard formula 
  current_f1 <- round((2 * current_precision * current_recall) / (current_precision + current_recall),2)
  
  #print out model performance if selected in function call
  if (print_out ==TRUE){
    #print out confusion matrix result
    print(confusion_matrix)
    cat("\n\n")
    #print out precision, recall and f1-score
    cat(paste("Accuracy: ",current_accuracy, "\nPrecision: ",current_precision, "\nRecall:",current_recall, "\nF1-score:",
            current_f1, "\nAUC: ", current_auc,  "\n\n", sep = " ") )
  }
  
  #if specified in function call to calculate ROC
  if (roc == TRUE){
    #make ROC plot squared without white padding
    par(pty = "s")
    #calculate Receiver operating characteristic with  area under the curve
    #change axis labels, print AUC on plot, specify title of plot
    roc(true_labels, predictions, plot=TRUE, legacy.axes=TRUE, percent=FALSE,
        xlab="False Positive Rate", ylab="True Postive Rate", col="#0da9d6", lwd=4, 
        print.auc=TRUE, main=title_plot)
  }
  metric_list <- list("precision" = current_precision, "recall" = current_recall, 
       "current_f1" = current_f1, "auc"=current_auc)
  
  #if no preinting of performance metrics is selected, function returns a list
  #of performance metrics
  if (print_out == FALSE) {
    metric_list <- list("accuracy"=current_accuracy, "precision"=current_precision, 
                        "recall"=current_recall, "f1"=current_f1, 
                        "auc"=current_auc)
    #function returns the metric list
    return(metric_list)
  }
}


# ******************************************************************************
#                     3.1 Dummy classifier
# ******************************************************************************

#define function that represents a dummy classifier which will be used as 
#baseline which uses a strategy of stratified guessing where prior probability of
#each class is used
dummy_classifier_stratified <- function(df, target_col){
  
  #if data stored as tibble transform into dataframe
  df <- as.data.frame(df)
  #number of predictions needed is number of rows in dataframe
  number_predictions <- nrow(df)
  
  #convert factor to numeric to calculate proportion of positive cases 
  #target_numeric <-  as.numeric(as.character(df[,target_col]))
  
  #get ratio of target class to derive probability for stratified strategy
  proportion_target_class <- sum(df[,target_col] == "deceased")/number_predictions
  
  #use binomial distribution with probability to return positive class based on 
  #ratio of positive to negative cases in dataframe
  set.seed(123)
  prediction_dummy <- rbinom(number_predictions, #number of observations
                             size=1, #how many trials are conducted
                             prob= proportion_target_class #probability of success per trial
  )
  #caret expects target column as factors and not numeric
  prediction_dummy <- as.factor(if_else(prediction_dummy == 1, "deceased", "survival"))
  
  #return predicted classes
  return (prediction_dummy)
}

#calculate predicted label
dummy_predict_label <- dummy_classifier_stratified(df = df_train, target_col = "deceased")

#receive evaluation metrics
dummy_metrics <- model_eval(predict_label=dummy_predict_label, true_labels=df_train$deceased,print_out =FALSE,
           predictions=FALSE)

#transform evaluation metrics into dataframe, rename columns
dummy_metric_df <- data.frame(list("predictor"="clinical & laboratory") ,dummy_metrics)
colnames(dummy_metric_df) <- c("Predictor variables",	"Accuracy",	"Precision",	"Recall",	"F1 score",	"AUC")

#write performance metrics to csv
write.csv(dummy_metric_df, "dummy_classifier_metrics.csv", row.names = F)

# ******************************************************************************
#                     3.2 Logistic regression
# ******************************************************************************

###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######

#using custom function model comparison and specifying generalized linear model
#family = "binomial" will be automatically assigned by the caret package as long as the outcome
#variable is a factor (was transformed to factor datatype previously)
glm_summary <- model_comparison(name_model="glm", name_model_output="Log. regression")

print(glm_summary)


# ******************************************************************************
#                     3.3 Penalized logistic regression: LASSO
# ******************************************************************************

#build custom function for lasso regression
#function is similar to function "model_comparison" but needs adjustments as
#lasso is not implemented in caret 
lasso_model_comparison <- function(){
  #initialize variables that will be used for summary dataframe
  current_predictors_range <- NULL
  accuracy_range <- NULL
  precision_range <- NULL
  recall_range <- NULL
  f1_score_range <- NULL
  auc_range <- NULL
  
  #for loop that performs model training and evaluation on separate selection of predictors
  for (i in c(1:3)){
    #only use clinical predictors 
    if (i == 1){
      current_predictors <- "clinical"
      training_data <- df_train %>% select(clinical_cols, deceased)
      testing_data <- df_test %>% select(clinical_cols, deceased)
    }
    #only use laboratory predictors 
    else if (i == 2){
      current_predictors <- "laboratory"
      training_data <- df_train %>% select(laboratory_cols, deceased)
      testing_data <- df_test %>% select(laboratory_cols, deceased)
    }
    #use full dataset for predictions 
    else {
      current_predictors <- "clinical & laboratory"
      training_data <- df_train 
      testing_data <- df_test 
    }
    
    #transform data format as cv.glmnet function expects data in design matrix format
    #for X matrix exclude target vector which is instead assigned to y
    X_train <-model.matrix(deceased~., training_data)[,-1]
    y_train <- training_data$deceased
    X_test <- model.matrix(deceased ~., testing_data)[,-1]
    
    #crossvalidation is used to optimize the lambda hyperparameter (amount of coefficient shrinkage)
    cv_lasso_model <- cv.glmnet(x = X_train, y = y_train, 
                                alpha = 1, #alpha of 1 equals lasso regression, alpha of 0 equals ridge regression
                                family = "binomial") #for binary response
    
    #Build model with lambda value that minimizes mean cross-validated error 
    lasso_model <- glmnet(X_train, y_train, family = "binomial", alpha = 1,
                          lambda = cv_lasso_model$lambda.1se) #result from CV
    
    #derive probabilities of test set
    lasso_model_prob <- predict(object=lasso_model,newx=X_test,
                                type='response')
    
    #transform probabilities into strings and then factors
    #use threshold of a probability of 0.5 to assign class labels
    lasso_model_pred_label <- as.factor(ifelse(lasso_model_prob > 0.5, "survival", "deceased"))
    
    
    #call evaluation function to calculate and display model performance
    performance_metrics_model <- model_eval(predict_label=lasso_model_pred_label, true_labels=testing_data$deceased,
               predictions=lasso_model_prob, title_plot = paste("LASSO regression (",current_predictors,")",sep = ""),
               roc=TRUE)
    
    #unpacking model performance metrics that are stored in a list
    #building up numeric vectors that can subsequently be used to build a summary
    #dataframe
    current_predictors_range[i] <- current_predictors
    accuracy_range[i] <- performance_metrics_model$accuracy
    precision_range[i] <- performance_metrics_model$precision
    recall_range[i] <- performance_metrics_model$recall
    f1_score_range[i] <- performance_metrics_model$f1
    auc_range[i] <- performance_metrics_model$auc
  }
  #build summary dataframe 
  summary_dataframe <- data.frame(current_predictors_range, accuracy_range, precision_range,
                                  recall_range, f1_score_range, auc_range)
  #change column names to better readable names
  colnames(summary_dataframe) <- c("Predictor variables", "Accuracy", "Precision",
                                   "Recall", "F1 score", "AUC")
  
  #write model performance to csv
  write.csv(summary_dataframe, "lasso_logistic_regression_metrics.csv", row.names = F)
  
  return (summary_dataframe)
}


###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######

#using custom function for different predictor variables for LASSO regression
lasso_summary <- lasso_model_comparison()

#print model summary
print(lasso_summary)


# ******************************************************************************
#                     3.2 K nearest neighbors
# ******************************************************************************

#using custom function for model comparison and specifying KNN as model
knn_summary <- model_comparison(name_model="knn", name_model_output="k-nearest neighbor")

#print model summary
print(knn_summary)


# ******************************************************************************
#                     3.2 Support vector machine with radial kernel
# ******************************************************************************

###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######

#using custom function model comparison and specifying radial kernel support 
#vector machine  
svm_summary <- model_comparison(name_model="svmRadial", name_model_output="Radial kernel SVM")

#print model summary
print(svm_summary)


# ******************************************************************************
#                     3.2 Gradient boosting machine
# ******************************************************************************

###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######

#using custom function model comparison and specifying radial kernel support 
#vector machine  
gbm_summary <- model_comparison(name_model="gbm", name_model_output="Gradient boosting machine")

#print model summary
print(gbm_summary)

# ******************************************************************************
#                     3.2 Random forest
# ******************************************************************************

###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######

#using custom function for model comparison and specifying random forest as model
rf_summary <- model_comparison(name_model="rf", name_model_output="Random forest")

#print model summary
print(rf_summary)


# ******************************************************************************
#                     3.2 Neural network
# ******************************************************************************

#build custom function for neural networks
#function is similar to function "model_comparison" but needs adjustments as
#neuralnetwork needs specific formula and data input
neural_network_model_comparison <- function(){
  #initialize variables that will be used for summary dataframe
  current_predictors_range <- NULL
  accuracy_range <- NULL
  precision_range <- NULL
  recall_range <- NULL
  f1_score_range <- NULL
  auc_range <- NULL
  
  #for loop that performs model training and evaluation on separate selection of predictors
  for (i in c(1:3)){
    #only use clinical predictors 
    if (i == 1){
      current_predictors <- "clinical"
      training_data <- df_train %>% select(clinical_cols, deceased)
      testing_data <- df_test %>% select(clinical_cols)
    }
    #only use laboratory predictors 
    else if (i == 2){
      current_predictors <- "laboratory"
      training_data <- df_train %>% select(laboratory_cols, deceased)
      testing_data <- df_test %>% select(laboratory_cols)
    }
    #use full dataset for predictions 
    else {
      current_predictors <- "clinical & laboratory"
      training_data <- df_train 
      testing_data <- df_test %>% select(-deceased)
    }
    #define target vector
    testing_target <- df_test$deceased
    
    #neural network function requires data in design matrix format and categorical 
    #variables must be encoded as dummy variables and all data must be numerical
    df_train_dummy <-model.matrix(~., df_train)[,-1]
    df_test_dummy <-model.matrix(~., df_test)[,-1]
    
    #get all column names of dataset
    colnames_nn_formula <- colnames(df_train_dummy)
    #build prediction formula for outcome deceased/survival by using all column names
    nn_formula <- as.formula(paste("deceasedsurvival ~", 
                                   paste(colnames_nn_formula[!colnames_nn_formula %in% "deceasedsurvival"], collapse = " + ")))
    
    #building neural network model
    nn_model <- neuralnet::neuralnet(formula = nn_formula, data=df_train_dummy, 
                                     hidden=c(32,12), #number of neurons in hidden layer
                                     act.fct = "logistic", #activation function for final layer - needed for classification
                                     linear.output = FALSE) #classification task, therefore model should output probabilities
    
    #get predictions of neural network
    nn_predictions <- neuralnet::compute(nn_model,df_test_dummy)
    
    #transform probabilities into strings and then factors
    #use threshold of a probability of 0.5 to assign class labels
    nn_predict_label <- as.factor(ifelse(nn_predictions$net.result > 0.5, "survival", "deceased"))
    
    #call evaluation function to calculate and display model performance
    performance_metrics_model <- model_eval(predict_label=nn_predict_label, 
                                            true_labels=testing_target,
                                            predictions=nn_predictions$net.result, 
                                            title_plot = paste("Neural network (",current_predictors,")",sep = ""),
                                            roc=TRUE)
    
    #unpacking model performance metrics that are stored in a list
    #building up numeric vectors that can subsequently be used to build a summary
    #dataframe
    current_predictors_range[i] <- current_predictors
    accuracy_range[i] <- performance_metrics_model$accuracy
    precision_range[i] <- performance_metrics_model$precision
    recall_range[i] <- performance_metrics_model$recall
    f1_score_range[i] <- performance_metrics_model$f1
    auc_range[i] <- performance_metrics_model$auc
  }
  #build summary dataframe 
  summary_dataframe <- data.frame(current_predictors_range, accuracy_range, precision_range,
                                  recall_range, f1_score_range, auc_range)
  #change column names to better readable names
  colnames(summary_dataframe) <- c("Predictor variables", "Accuracy", "Precision",
                                   "Recall", "F1 score", "AUC")
  #printing model summary
  cat(paste("Performance for the following model: Neural network", "\n\n",sep=""))
  
  
  #write model performance to csv
  write.csv(summary_dataframe, "neural_net_metrics.csv", row.names = F)
return(summary_dataframe)
  }


###### CAUTION - COMPUTATIONALLY INTENSIVE STEP ######
nn_summary <- neural_network_model_comparison()

#print model summary
print(nn_summary)



# ******************************************************************************
#                     3.2 Ensemble classifier with averaging
# ******************************************************************************

#define all models that are part of ensemble model
ensemble_vector <- c("rf","glm", "svmRadial", "glm", "knn")

###### CAUTION - COMPUTATIONALLY INTENSIVE STEPS ######

#train all models that are subsequently used as an ensemble
ensemble_models <- caretList(deceased ~ ., data=df_train, 
                             methodList=ensemble_vector, 
                             metric="ROC",                  #metric for optimization  
                             trControl=train_control_cv) #use cross validation

#training ensemble model
ensemble_glm <- caretStack(all.models=ensemble_models, method="glm", metric="ROC", 
                           trControl= train_control_cv) #see above

#average ensemble predictions by using a linear classifier
ensemble_predictions <- predict(ensemble_glm, newdata=df_test)
          
#print out evaluation metrics and confusion matrix
ensemble_metrics <-  model_eval(predict_label=ensemble_predictions, 
                     true_labels=df_test$deceased,
                     print_out =FALSE,
                     predictions=FALSE)

#transform evaluation metrics into dataframe, rename columns
ensemble_metric_df <- data.frame(list("predictor"="clinical & laboratory"), ensemble_metrics)
colnames(ensemble_metric_df) <- c("Predictor variables",	"Accuracy",	"Precision",	"Recall",	"F1 score",	"AUC")

#write performance metrics to csv
write.csv(ensemble_metric_df, "ensemble_classifier_metrics.csv", row.names = F)

#print model summary
print(ensemble_metric_df)


# ******************************************************************************
#                     3.2 Visualization of model importance
# ******************************************************************************

#adding model as new column to each dataframe that stores performance of 
#individual models
glm_summary <- add_column(glm_summary, model = rep(c("log regression"),dim(glm_summary)[1]), 
                          .before = "Predictor variables")

lasso_summary <- add_column(lasso_summary, model = rep(c("Lasso regression"),dim(lasso_summary)[1]), 
                          .before = "Predictor variables")

knn_summary <- add_column(knn_summary, model = rep(c("KNN"),dim(knn_summary)[1]), 
                            .before = "Predictor variables")

gbm_summary <- add_column(gbm_summary, model = rep(c("GBM"),dim(gbm_summary)[1]), 
                          .before = "Predictor variables")

svm_summary <- add_column(svm_summary, model = rep(c("GBM"),dim(svm_summary)[1]), 
                          .before = "Predictor variables")

rf_summary <- add_column(rf_summary, model = rep(c("RF"),dim(rf_summary)[1]), 
                          .before = "Predictor variables")

nn_summary <- add_column(nn_summary, model = rep(c("N Net"),dim(nn_summary)[1]), 
                         .before = "Predictor variables")

ensemble_metric_df <- add_column(ensemble_metric_df, model = rep(c("Ensemble models"),dim(ensemble_metric_df)[1]), 
                         .before = "Predictor variables")

#combining all different summary metrics into a single dataframe
total_model_summary <- rbind(
glm_summary,
lasso_summary,
knn_summary,
svm_summary,
gbm_summary,
rf_summary,
nn_summary,
ensemble_metric_df)

#only select relevant columns
total_model_summary <- total_model_summary %>% select(model, "Predictor variables", "F1 score")

#standardize column names for plotting function
names(total_model_summary) <- make.names(names(total_model_summary))

#plot comparison of different model performance as barplot
ggplot(total_model_summary) +
 aes(x = model, weight = F1.score) +
 geom_bar(fill = "#112446") +
 facet_wrap(vars(Predictor.variables)) +
 labs(title = "Comparison of model performance",x="Classifier", 
      y = "f1-score", caption = "Missing bars if metric is not available.") +
theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))




# ******************************************************************************
#                    4. Clearing environment
# ******************************************************************************

#clearing environment
rm(list=ls())