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
set.seed(42)

# 2) set path to that folder where R script is located, then access data folder 
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
wd_path <- "./data/"
setwd(wd_path)

# 3) installing (if not present) and loading packages

#Required packages: "useful","tidyverse", "patchwork", "mice", "tibble", "ggplot2"

package_checker(c("caret",
                  "readr",     #read in csv files
                  "tidyverse", #data wrangling
                  "patchwork",  #to align multiple plots
                  "useful",    #decision of number of clusters 
                  "tibble",
                  "ggplot2",
                  "kernlab",  #support vector machines
                  #"shapper",
                  #"shapr",
                  "randomForest",
                  #"caTools",  # calculates AUC
                  "ggcorrplot",
                  "glmnet", #lasso logistic regression
                  "pROC" ,#function for AUC ROC, precision, recall, confusion matrix
                  "gbm", #gradient boosting machines
                  "neuralnet",  #neural networks
                  "class" #KNN
))

#if the custom function "package_checker" does not work (although this was 
#tested on different computers), please uncomment the following and run it:
# install.packages("tidyverse")
# install.packages("patchwork")
# install.packages("mice")
# install.packages("tibble")
# install.packages("ggplot2")
# library("tidyverse")
# library("patchwork")
# library("mice")
# library("tibble")
# library("ggplot2")
#library("gbm")

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

#date of hospital admission, referral from physician and patient ID have no further
#predictive value
df <- subset(df, select = c(-PatID,-date_admission_hospital, -referral))

#organ replacement with ECMO (artifical lung) and dialysis(artificial kidney), 
#mechanical ventilation (intubation) all only occur after admission 
#to avoid information leak from the future removing these variables
#leave "intubated_referral" in dataset as this is known at admission
df <- subset(df, select = c(-ecmo_treatment,-dialysis_treatment, -intubation
                            ))

factor_cols <-  c("hypertension",
                  "cardiovascular_disease",
                  "asthma",
                  "copd",
                  "diabetes",
                  "diabetes_complication",
                  "hypercholesterolemia",        
                  "kidney_disease"       ,        "dialysis"             ,        "liver_disease"          ,     
                  "neurological_disease"  ,       "stroke"                ,       "immunosuppression"       ,    
                  "smoker"                 ,      "intubated_referral"     ,      "deceased"                 ,   
                  "dnr_dni"                 ,     "female"                  ,     "dexa_treatment",
                  "dexa_standard_period"
)

#using vectorized function to change the column type
df[factor_cols] <- lapply(df[factor_cols], factor)



# ******************************************************************************
#                     3 Model building
#                     3.1 Train/Test Split and data normalization
# ******************************************************************************


#get random stratified sample (stratified by outcome)
#get row numbers for training data
set.seed(42)
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
dummy_encoding_train <- dummyVars(deceased ~ ., data = df_train, fullRank=T)
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

##TO DO::::::

#defining columns that contain clinical or laboratory data

#use colname function to retrieve all column names, using vector indexing to only
#get clinical variables (that come first in order)
clinical_cols <- colnames(df_train)[0:22] 

#must append two clinical variables that are located at different positions in 
#original vector
clinical_cols <- append(clinical_cols, c("bmi", "dexa_standard_period","dexa_standard_period.1"))

#verify that all clinical variables were retrieved
print(clinical_cols)

#select all columns as laboratory variables except clinical variables and deceased 
#which is the outcome (target)
laboratory_cols <- df_train %>% select(-one_of(append(clinical_cols, "deceased"))) %>% colnames()

#verify that all laboratory variables are present
print(laboratory_cols)


#define function to evaluate models with confusion matrix and if specified also
#calculate area under curve
model_eval <- function(predict_label, true_labels, predictions, title_plot="", roc=FALSE){
  
  true_labels <- as.factor(true_labels)
  
  #calculate confusion matrix, function requires factor columns as input
  conf_matrix <- confusionMatrix(data=predict_label, reference = factor(true_labels),
                                 mode = "everything", positive="deceased")
  #print out confusion matrix result
  print(conf_matrix)
  
  # precision <- posPredValue(predict_label, true_labels, positive="1")
  # recall <- sensitivity(predict_label, true_labels, positive="1")
  # 
  # f1_score <- (precision * recall * 2) / (precision + recall)
  # 
  # print(paste("F1 Score is: ",f1_score), sep="")
  
  ## ROC for model
  # plot_roc_curve(obese, model, plot=TRUE, legacy.axes=TRUE, 
  #     percent=FALSE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#4daf4a", lwd=4, print.auc=TRUE)
  #https://github.com/StatQuest/roc_and_auc_demo/blob/master/roc_and_auc_demo.R
  
  #if specified in function call to calculate ROC
  if (roc == TRUE){
    #to make ROC plot squared without white padding
    par(pty = "s")
    #calculate Receiver operating characteristic with  area under the curve
    #change axis labels, print AUC on plot, specify title of plot
    roc(true_labels, predictions, plot=TRUE, legacy.axes=TRUE, percent=FALSE,
        xlab="False Positive Rate", ylab="True Postive Rate", col="#288BA8", lwd=4, 
        print.auc=TRUE, main=title_plot)
    
    #legend("bottomright", legend="test", col=c("#288BA8"), lwd=4)
  }
  
}

# ******************************************************************************
#                     3.1 Dummy classifier
# ******************************************************************************

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
  prediction_dummy <- rbinom(number_predictions,
                             size=1, 
                             prob= proportion_target_class
  )
  prediction_dummy <- if_else(prediction_dummy == 1, "deceased", "survival")
  
  prediction_dummy <- as.factor(prediction_dummy)
  
  #return predicted classes
  return (prediction_dummy)
}

#calculate predicted label
dummy_predict_label <- dummy_classifier_stratified(df = df_train, target_col = "deceased")

model_eval(predict_label=dummy_predict_label, true_labels=df_train$deceased,
           predictions=FALSE)



# ******************************************************************************
#                     3.2 Logistic regression
# ******************************************************************************



train_control_cv <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 6,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 


#logistic regression
# lr_model <- glm(formula=deceased ~ . , family = binomial(link='logit'),
#                 data = df_train)

#train logistic regression model
lr_model<- train(form=deceased~., data=df_train, trControl=train_control_cv, 
                 method="glm", family = binomial(link='logit'))

#print model specifications
summary(lr_model)

#calculate predictions
lr_model_prob <- predict(object=lr_model,newdata=df_test,type='prob')$survival
#lr_model_predict_labels <- lr_model_prob

#use calculated probabilities to assign class label based on 0.5 threshold
lr_model_predict_labels <- as.factor(ifelse(lr_model_prob > 0.5,"survival","deceased"))



#print results and display ROC of model
model_eval(predict_label=lr_model_predict_labels, true_labels=df_test$deceased,
           title_plot="Simple logistic regression" , predictions=lr_model_prob, roc=TRUE)


# ******************************************************************************
#                     3.3 Penalized logistic regression: LASSO
# ******************************************************************************


#transform data format as cv.glmnet function expects data in design matrix format
#for X matrix exclude target vector which is instead assigned to y
X_train <-model.matrix(deceased~., df_train)[,-1]
y_train <- df_train$deceased
X_test <- model.matrix(deceased ~., df_test)[,-1]

#crossvalidation is used to optimize the lambda hyperparameter (amount of coefficient shrinkage)
cv_lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")

#Build model with lambda value that minimizes mean cross-validated error 
lasso_model <- glmnet(X_train, y_train, family = "binomial", alpha = 1, 
                lambda = cv_lasso_model$lambda.min)

#Show coefficients of LASSO model
coef(lasso_model)

#derive probabilities of test set
lasso_model_prob <- predict(object=lr_model,newdata=df_test %>% select(-deceased),
                            type='prob')$survival

#transform probabilites into strings and then factors
lasso_model_pred_label <- as.factor(ifelse(lasso_model_prob > 0.5, "survival", "deceased"))

#evaluation function 
model_eval(predict_label=lasso_model_pred_label, true_labels=df_test$deceased,
           predictions=lasso_model_prob, title_plot="Penalized logistic regression (LASSO)",
           roc=TRUE)


# ******************************************************************************
#                     3.2 Support vector machine
# ******************************************************************************

#build radial SVM model 
svm_radial_model = train(deceased ~ ., data=df_train %>% clinical_cols, method='svmRadial', 
                        trControl = train_control_cv)

#print model specifications
svm_radial_model

#calculate predictions
svm_radial_model_prob <- predict(object=svm_radial_model,newdata=df_test,type='prob')$survival
#lr_model_predict_labels <- lr_model_prob

#use calculated probabilities to assign class label based on 0.5 threshold
svm_radial_model_predicted_labels <- as.factor(ifelse(svm_radial_model_prob > 0.5,"survival","deceased"))



#print results and display ROC of model
model_eval(predict_label=svm_radial_model_predicted_labels, true_labels=df_test$deceased,
           title_plot="Radial kernel support vector machine" , predictions=svm_radial_model_prob, roc=TRUE)


# ******************************************************************************
#                     3.2 Gradient boosting machine
# ******************************************************************************

#establish gradient boosting model
gbm_model <- train(deceased ~ ., data = df_train, 
                 method = "gbm", trControl = train_control_cv,
                 verbose = F)

#print model specifications
gbm_model

#calculate predictions
gbm_model_prob <- predict(object=gbm_model,newdata=df_test,type='prob')$survival
#lr_model_predict_labels <- lr_model_prob

#use calculated probabilities to assign class label based on 0.5 threshold
gbm_predicted_labels <- as.factor(ifelse(gbm_model_prob > 0.5,"survival","deceased"))



#print results and display ROC of model
model_eval(predict_label=gbm_predicted_labels, true_labels=df_test$deceased,
           title_plot="Gradient boosting machine" , predictions=gbm_model_prob, roc=T)


# ******************************************************************************
#                     3.2 Random forest
# ******************************************************************************


rf_model <- randomForest(formula=deceased ~ . , data = df_train)

print(rf_model)

varImpPlot(rf_model,type=2)

#predict test set
rf_predictions <- predict(rf_model, df_test %>% select(-deceased), type="prob")[,1]

rf_predict_label <- ifelse(rf_predictions >0.5,0,1 )

model_eval(predict_label=rf_predict_label, true_labels=df_test$deceased,
           predictions=rf_predictions, roc=TRUE)


# ******************************************************************************
#                     3.2 K nearest neighbors
# ******************************************************************************

knn_predict_label <- knn(train = df_train %>% select(-deceased), 
                       test =df_test %>% select(-deceased), df_train$deceased, k=3, prob=FALSE)

#knn_predict_label <- knn_predictions

model_eval(predict_label=knn_predict_label, true_labels=df_test$deceased,
           predictions=None, roc=FALSE)

table(df_test$deceased != knn_predict_label)

predicted.purchase = NULL
error.rate = NULL

for(i in 1:20){
  set.seed(42)
  predicted.purchase = knn(train.data,test.data,train.purchase,k=i)
  error.rate[i] = mean(test.purchase != predicted.purchase)
}

k.values <- 1:20

error.df <- data.frame(error.rate,k.values)

ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point()+ 
  geom_line(lty="dotted",color='red')










# ******************************************************************************
#                     3.2 Neural networks
# ******************************************************************************


#neural network function requires data in design matrix format and categorical 
#variables must be encoded as dummy variables
df_train_dummy <-model.matrix(~., df_train)[,-1]


df_test_dummy <-model.matrix(~., df_test)[,-1]

#n <- names(df_train_dummy)

n <- colnames(df_train_dummy)#names(df_train)

f <- as.formula(paste("deceased1 ~", paste(n[!n %in% "deceased1"], collapse = " + ")))

#https://stackoverflow.com/questions/17457028/working-with-neuralnet-in-r-for-the-first-time-get-requires-numeric-complex-ma
#NN requires factors into dummy encoding

####COMPUTATION HEAVY

nn <- neuralnet::neuralnet(formula = f, data=df_train_dummy, hidden=c(5,3), 
                           linear.output=TRUE)

#get predictions of neural network
nn_predictions <- neuralnet::compute(nn,df_test_dummy)

#use threshold of a probability of 0.5 to assign class labels
nn_predict_label <-  ifelse(predicted.nn.values$net.result >0.5,1,0)

#visualize model performance with AUC and confusion matrix
model_eval(predict_label = nn_predict_label,true_labels =df_test$deceased ,predictions = nn_predictions, 
           roc = TRUE)






distances <- dist(df)

clustering <- hclust(df)
plot(clustering, main="")



















# ******************************************************************************
#                    4. Saving processed data and clearing environment
# ******************************************************************************

# #creating name for file to save that contains current date
# name_of_file <- paste(Sys.Date(),"predictions.csv", sep = "")
# 
# #saving dataframe as CSV without row names
# write.csv(x=df_clean, file=name_of_file, row.names = FALSE)


#clearing environment
rm(list=ls())



