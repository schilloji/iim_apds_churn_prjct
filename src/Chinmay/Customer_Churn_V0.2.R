install.packages("neuralnet",dependencies = TRUE)
install.packages("MASS",dependencies = TRUE)
install.packages("kerasR",dependencies = TRUE)
install.packages("onehot", dependencies = TRUE)
install.packages("mltools", dependencies = TRUE)
install.packages("randomForest", dependencies = TRUE)
install.packages("plotROC", dependencies = TRUE)


library(MASS)
library(neuralnet)
library(kerasR)
library(onehot)
library(caret)
library(mltools)
library(data.table)
library(ROCR)
library("rpart")
library("rpart.plot")
library(randomForest)
library(pROC)
library(plotROC)


#setwd("path") ### set working directory

##### set parameters ###################################################################################################

options(max.print=999999)



######## reading the file post-datacleaning ############################################################################
churn_data <- read.csv("prep_telo_df.csv")
summary(churn_data)

###### Check for the missing values ###################

columns_NA<-apply(churn_data,2,function(x) sum(is.na(x)))
columns_NA_SRT<-sort(columns_NA, decreasing = TRUE)
columns_NA_SRT
sum(columns_NA_SRT > 0) ##### number of columns having missing values
columns_NA_SRT_pct
data_NA <-sum(is.na(churn_data))
data_NA

####### Data imputation: No data imputation as the input data has already been  cleaned through data preparation steps###
churn_data_cln<-churn_data
#########################################################################################################################

churn_data_cln$Customer_ID<-NULL ### remove customer id
churn_data_cln1 <- churn_data_cln[, which(colSums(churn_data_cln) > 0) ] 
head(churn_data_cln1,5)
# Check that there is  no data with  missing value
apply(churn_data_cln1,2,function(x) sum(is.na(x)))

##### Scaling of data ##################################################################################################
maxs <- apply(churn_data_cln1, 2, max) 
mins <- apply(churn_data_cln1, 2, min)
scaled_cln <- as.data.frame(scale(churn_data_cln1, center = mins, scale = maxs - mins))

#########end: scaling of data ##########################################################################################


########################################## splitting between train and the test data ###################################
# Train-test random splitting for linear model
index <- sample(1:nrow(scaled_cln),round(0.7*nrow(scaled_cln))) # splittig train:test as 70:30 .
train_cln <- scaled_cln[index,]
test_cln <- scaled_cln[-index,]

########################## End: splitting between train and the test data #############################################


#######  Logistic Regression  with all parameters #####################################################################

ChurnLogit<-glm(train_cln$churn~.,data=train_cln,
                     family = binomial("logit"))
summary(ChurnLogit)

#prediction 
ChurnPrediction<-predict(ChurnLogit,test_cln)
summary(ChurnPrediction)
ChurnPrediction
ChurnPrediction_resp<-predict(ChurnLogit,test_cln,
                              type="response")

ChurnPrediction_resp

ChurnPrediction_prob_glm<-prediction(ChurnPrediction_resp,
                                     test_cln$churn)

ChurnPrediction_prob_glm
#create the performance object
churn_perf_glm<-performance(ChurnPrediction_prob_glm,"tpr","fpr") 

#Plot function to plot the ROC curve

plot(churn_perf_glm)


# Get the Area under the ROC curve (AOC)
auc_ROCR <- performance(ChurnPrediction_prob_glm, measure = "auc")
auc_ROCR
auc_ROCR@y.values[[1]] 

################################## End Logistic Regression with all parameters ########################################

######################## Neutal network with  relevent fields ############################################################

f1_nn <- churn ~ rev_Mean + totmrc_Mean + ovrmou_Mean + roam_Mean + drop_vce_Mean + threeway_Mean + iwylis_vce_Mean + months + uniqsubs + totcalls + hnd_price + phones + lor + eqpdays + fe_mean_per_minute_charge + fe_tot_revenue_per_call + fe_tot_mou_per_call + fe_tot_revenue_adj + asl_flag_N + crclscod_BA + crclscod_EA + crclscod_ZA + ethnic_O  + kid16_17_U  + prizm_social_one_Missing + prizm_social_one_R + prizm_social_one_T

f1_nn
#create the neural network. Architecture - two hidden layers with 10 and 3 neurons respectively. 
nn <- neuralnet(f1_nn,data=train_cln,hidden=c(10,5,3), threshold = 0.5)


help(neuralnet) #read the help file for neuralnet.

# Visual plot of the model

plot(nn)

### result set out of the training

nn$model.list
nn$linear.output
nn$result.matrix

churn_test_prob = neuralnet::compute(nn,test_cln)

churn_test_prob_result<-churn_test_prob$net.result

prediction_nn <-ifelse(churn_test_prob_result>0.5,1,0)
actual_nn<-test_cln$churn

#create a dataframe with the actual and the predicted values
resultsdf_nn<-data.frame(actual=actual_nn,prediction=prediction_nn[,1])

##### confusin matrix
Churn_Conf_matrix_nn<-confusionMatrix(factor(resultsdf_nn$prediction),
                                      reference=factor(test_cln$churn))
Churn_Conf_matrix_nn
### Create the ROC########################################################

churn_nn_pred = prediction(churn_test_prob_result, test_cln$churn)
perf_nn_churn <- performance(churn_nn_pred, "tpr", "fpr")
plot(perf_nn_churn, colorize = TRUE)
abline(a=0, b=1)

##### AUC of the ROC curve ################################### 

auc_ROCR_nn <- performance(churn_nn_pred, measure = "auc")
auc_ROCR_nn
auc_ROCR_nn@y.values[[1]]

######################## End: Neutal network with  relevent fields ################################################################################


