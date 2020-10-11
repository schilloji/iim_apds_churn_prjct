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


#setwd("path") #### Setting the path

##### set parametesr ######################

options(max.print=999999)



######## reading the file post-datacleaning #########################################################################
churn_data <- read.csv("prep_telo_df.csv")
head(churn_data)

summary(churn_data)

###### check for the missing values ###################

columns_NA<-apply(churn_data,2,function(x) sum(is.na(x)))
columns_NA_SRT<-sort(columns_NA, decreasing = TRUE)
columns_NA_SRT
columns_NA_SRT_pct<-columns_NA_SRT/nrow(churn_data)
sum(columns_NA_SRT > 0) ##### number of columns having missing values
columns_NA_SRT_pct

class(columns_NA)
sort(apply(churn_data,2,function(x) sum(is.na(x))))

data_NA <-sum(is.na(churn_data))

data_NA

####### Data imputation##############################################################################

###### No data imputation is done as the input is the cleaned up data ###############################
churn_data_im<-churn_data
sum(is.na(churn_data_im))
#####################################################################################################

head(churn_data_im)


churn_data_cln<-churn_data_im

churn_data_cln$Customer_ID<-NULL ### remove customer id
churn_data_cln1 <- churn_data_cln[, which(colSums(churn_data_cln) > 0) ] ### remove columns that has sum <= zero

head(churn_data_cln1,5)
# Check that there is  no data with  missing value
apply(churn_data_cln1,2,function(x) sum(is.na(x)))

##### Scaling of data ########################################################################################

# Scaling data for the NN
maxs <- apply(churn_data_cln1, 2, max) 
mins <- apply(churn_data_cln1, 2, min)
scaled_cln <- as.data.frame(scale(churn_data_cln1, center = mins, scale = maxs - mins))


head(scaled_cln,5)

############################ end Data cleaning##################################################################


########################################## splitting between train and the test data ###########################
# Train-test random splitting for linear model
index <- sample(1:nrow(scaled_cln),round(0.7*nrow(scaled_cln))) # splittig train:test as 70:30 .
train_cln <- scaled_cln[index,]
test_cln <- scaled_cln[-index,]

########################## End: splitting between train and the test data ########################################


#######  Logistic Regression  with all parameters ################################################################

ncol(train_cln)

ChurnLogit<-glm(train_cln$churn~.,data=train_cln,
                     family = binomial("logit"))
summary(ChurnLogit)


write.csv(ChurnLogit$effects, "churnlogit.csv")

#prediction 
ChurnPrediction<-predict(ChurnLogit,test_cln)
summary(ChurnPrediction)
ChurnPrediction
ChurnPrediction_prob<-predict(ChurnLogit,test_cln,
                              type="response")

ChurnPrediction_prob

ChurnPrediction_prob_glm<-prediction(ChurnPrediction_prob,
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

######## Begin: Random Forest with selected fields ###########################################################################################

#f1_rf <- churn ~ rev_Mean + mou_Mean + totmrc_Mean + custcare_Mean + ccrndmou_Mean + cc_mou_Mean + threeway_Mean + months + uniqsubs + actvsubs + asl_flag_N + avgmou + avg3mou + avg3rev + prizm_social_one_ + prizm_social_one_R + prizm_social_one_T + area_CALIFORNIA.NORTH.AREA  + area_NEW.ENGLAND.AREA + area_NEW.YORK.CITY.AREA + area_NORTH.FLORIDA.AREA + area_NORTHWEST.ROCKY.MOUNTAIN.AREA + area_PHILADELPHIA.AREA + area_SOUTH.FLORIDA.AREA + refurb_new_N + hnd_price + lor + HHstatin_ + ethnic_B + ethnic_D + ethnic_F + ethnic_G + ethnic_H + ethnic_I + ethnic_J + ethnic_N + ethnic_O + ethnic_R + ethnic_S + ethnic_U + kid16_17_U + eqpdays

#f1_rf
#Churn_random_forest<- randomForest(f1_rf, data = train_cln, ntree = 100, mtry = 3, importance = TRUE)
#churn_predict.rf=predict(Churn_random_forest,test_cln,type="response")

#ChurnPredictionReduced_prob_rf<-prediction(churn_predict.rf,
#                                            test_cln$churn)
#churn_perf_rf<-performance(ChurnPredictionReduced_prob_rf,"tpr","fpr") 
#plot(churn_perf_rf)

#auc_ROCR_rf <- performance(ChurnPredictionReduced_prob_rf, measure = "auc")
#auc_ROCR_rf@y.values[[1]] 

#?randomForest

######## End: Random Forest with selected fields ###########################################################################################

######## Begin: Random Forest with all fields fields ###########################################################################################

#f1_rf <- churn ~ rev_Mean + mou_Mean + totmrc_Mean + custcare_Mean + ccrndmou_Mean + cc_mou_Mean + threeway_Mean + months + uniqsubs + actvsubs + asl_flag_N + avgmou + avg3mou + avg3rev + prizm_social_one_ + prizm_social_one_R + prizm_social_one_T + area_CALIFORNIA.NORTH.AREA  + area_NEW.ENGLAND.AREA + area_NEW.YORK.CITY.AREA + area_NORTH.FLORIDA.AREA + area_NORTHWEST.ROCKY.MOUNTAIN.AREA + area_PHILADELPHIA.AREA + area_SOUTH.FLORIDA.AREA + refurb_new_N + hnd_price + lor + HHstatin_ + ethnic_B + ethnic_D + ethnic_F + ethnic_G + ethnic_H + ethnic_I + ethnic_J + ethnic_N + ethnic_O + ethnic_R + ethnic_S + ethnic_U + kid16_17_U + eqpdays

#f1_rf
#Churn_random_forest<- randomForest(churn ~., data = train_cln, mtry = 10, importance = TRUE)

#churn_predict.rf=predict(Churn_random_forest,test_cln,type="response")

#ChurnPredictionReduced_prob_rf<-prediction(churn_predict.rf,
#                                           test_cln$churn)
#churn_perf_rf<-performance(ChurnPredictionReduced_prob_rf,"tpr","fpr") 
#plot(churn_perf_rf)

#auc_ROCR_rf <- performance(ChurnPredictionReduced_prob_rf, measure = "auc")
#auc_ROCR_rf@y.values[[1]] 

#?randomForest

######## End: Random Forest with selected fields ###########################################################################################




######################## Neutal network with  relevent fields ############################################################
#head(scaled_cln,5)
### the formula
f1_nn <- churn ~ rev_Mean + totmrc_Mean + ovrmou_Mean + roam_Mean + drop_vce_Mean + threeway_Mean + iwylis_vce_Mean + months + uniqsubs + totcalls + hnd_price + phones + lor + eqpdays + fe_mean_per_minute_charge + fe_tot_revenue_per_call + fe_tot_mou_per_call + fe_tot_revenue_adj + asl_flag_N + crclscod_BA + crclscod_EA + crclscod_ZA + ethnic_O  + kid16_17_U  + prizm_social_one_Missing + prizm_social_one_R + prizm_social_one_T

f1_nn
#create the neural network. Architecture - two hidden layers with 10 and 3 neurons respectively. 
nn <- neuralnet(f1_nn,data=train_cln,hidden=c(10,5,3), threshold = 0.5)


help(neuralnet) #read the help file for neuralnet.

# Visual plot of the model

plot(nn)

### result set out of training

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


