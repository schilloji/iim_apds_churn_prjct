install.packages("neuralnet",dependencies = TRUE)
install.packages("MASS",dependencies = TRUE)
install.packages("kerasR",dependencies = TRUE)
install.packages("onehot", dependencies = TRUE)
install.packages("mltools", dependencies = TRUE)


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



#setwd("path") ##### set path 

##### set parameters ######################

options(max.print=999999) 

# rmColNApct = 0.90 #### threhold to exclude a column

######## reading the file and data cleaning #########################################################################
churn_data <- read.csv("Telecom_customer churn.csv")
head(churn_data)

summary(churn_data)

###### check for the missing values #############

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

churn_data_1 <- churn_data


churn_data_im<-churn_data

head(churn_data_im)
churn_data_im$numbcars[is.na(churn_data_im$numbcars)]<-0 #### set missing values  = 0
churn_data_im$lor[is.na(churn_data_im$lor)] <- sample.int(1:15) #### set missing values randomly between 1 to 15
churn_data_im$income[is.na(churn_data_im$income)] <- sample(c(4,5,6,7,8), 1) ##### set missing values with values having highest frequencies
churn_data_im$adults[is.na(churn_data_im$adults)] <- sample(c(1,2),1) ##### set missing values with values having highest frequencies
churn_data_im$hnd_webcap[is.na(churn_data_im$hnd_webcap)] <-'WCMB' ##### set missing values with 'WCMB' as this has very high frequency
churn_data_im$avg6mou[is.na(churn_data_im$avg6mou)]<-sample(c(0,1,11,65,5),1) #### choose from values with highest frequncies
churn_data_im$avg6qty[is.na(churn_data_im$avg6qty)]<-sample(c(0,44,30,49,53),1) #### choose from values with highest frequencies
churn_data_im$avg6rev[is.na(churn_data_im$avg6rev)]<-round(mean(churn_data_im$avg6rev, na.rm = TRUE)) ### avergae value
churn_data_im$truck[is.na(churn_data_im$truck)] <-0 #### valuees with highest frequency
churn_data_im$rv[is.na(churn_data_im$rv)] <- 0 ###### value with highest frequency
churn_data_im$forgntvl[is.na(churn_data_im$forgntvl)] <-0 ###### value with highest frequency
churn_data_im$change_mou[is.na(churn_data_im$change_mou)] <-round(mean(churn_data_im$change_mou, na.rm = TRUE)) #### average value
churn_data_im$change_rev[is.na(churn_data_im$change_rev)] <-round(mean(churn_data_im$change_rev, na.rm = TRUE)) #### average value
churn_data_im$hnd_price[is.na(churn_data_im$hnd_price)] <-round(mean(churn_data_im$hnd_price, na.rm = TRUE)) #### average value
churn_data_im$rev_Mean[is.na(churn_data_im$rev_Mean )] <-round(mean(churn_data_im$rev_Mean, na.rm = TRUE)) #### average value
churn_data_im$mou_Mean[is.na(churn_data_im$mou_Mean )] <-round(mean(churn_data_im$mou_Mean, na.rm = TRUE)) #### average value
churn_data_im$totmrc_Mean[is.na(churn_data_im$totmrc_Mean)] <-round(mean(churn_data_im$totmrc_Mean, na.rm = TRUE)) #### average value
churn_data_im$da_Mean[is.na(churn_data_im$da_Mean)] <-round(mean(churn_data_im$da_Mean, na.rm = TRUE)) #### average value
churn_data_im$ovrmou_Mean[is.na(churn_data_im$ovrmou_Mean)] <-round(mean(churn_data_im$ovrmou_Mean, na.rm = TRUE)) #### average value
churn_data_im$ovrrev_Mean[is.na(churn_data_im$ovrrev_Mean)] <-round(mean(churn_data_im$ovrrev_Mean, na.rm = TRUE)) #### average value
churn_data_im$vceovr_Mean[is.na(churn_data_im$vceovr_Mean)] <-round(mean(churn_data_im$vceovr_Mean, na.rm = TRUE)) #### average value
churn_data_im$datovr_Mean[is.na(churn_data_im$datovr_Mean)] <-round(mean(churn_data_im$datovr_Mean, na.rm = TRUE)) #### average value
churn_data_im$roam_Mean[is.na(churn_data_im$roam_Mean)] <-round(mean(churn_data_im$roam_Mean, na.rm = TRUE)) #### average value
churn_data_im$phones[is.na(churn_data_im$phones)] <-1 #### highest frequency value
churn_data_im$models[is.na(churn_data_im$models)] <-1 #### highest frequency value
churn_data_im$eqpdays[is.na(churn_data_im$eqpdays)] <-310 #### highest frequency value

sum(is.na(churn_data_im))


head(churn_data_im)

dt<-data.table(churn_data_im)

churn_data_cl_one<-one_hot(dt, dropCols = TRUE)

head(churn_data_cl_one)
col(churn_data_cl_one)
ncol(churn_data_cl_one)
churn_data_cln <-data.frame(churn_data_cl_one)

str(churn_data_cln) ### now the data is one-hot encoded for categorical variables 
churn_data_cln$Customer_ID<-NULL ### remove customer id
churn_data_cln1 <- churn_data_cln[, which(colSums(churn_data_cln) > 0) ] ### remove columns that has sum zero

head(churn_data_cln1,5)
# Check that there is  no data with  missing value
apply(churn_data_cln1,2,function(x) sum(is.na(x)))

##### Scaling of data ########################################################################################

maxs <- apply(churn_data_cln1, 2, max) 
mins <- apply(churn_data_cln1, 2, min)
scaled_cln <- as.data.frame(scale(churn_data_cln1, center = mins, scale = maxs - mins))


head(scaled_cln,5)

############################ end Data cleaning##################################################################

########################################## Begin: splitting between train and the test data ###########################
# Train-test random splitting 
index <- sample(1:nrow(scaled_cln),round(0.7*nrow(scaled_cln))) # splittig train:test with 70:30 .
train_cln <- scaled_cln[index,]
test_cln <- scaled_cln[-index,]

########################## End: splitting between train and the test data ########################################

#######  Begin: Logistic Regression  with all parameters ################################################################

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

#The plot function would now plot the ROC curve

plot(churn_perf_glm)


# Get the Area under the ROC curve (AOC)
auc_ROCR <- performance(ChurnPrediction_prob_glm, measure = "auc")
auc_ROCR
auc_ROCR@y.values[[1]] 
####### End: Logistic Regression  with all parameters ################################################################

######################## Begin: Neutal network with  relevent fields ############################################################

f1_nn <- churn ~ rev_Mean + mou_Mean + totmrc_Mean + custcare_Mean + ccrndmou_Mean + cc_mou_Mean + threeway_Mean + months + uniqsubs + actvsubs + asl_flag_N + avgmou + avg3mou + avg3rev + prizm_social_one_ + prizm_social_one_R + prizm_social_one_T + area_CALIFORNIA.NORTH.AREA  + area_NEW.ENGLAND.AREA + area_NEW.YORK.CITY.AREA + area_NORTH.FLORIDA.AREA + area_NORTHWEST.ROCKY.MOUNTAIN.AREA + area_PHILADELPHIA.AREA + area_SOUTH.FLORIDA.AREA + refurb_new_N + hnd_price + lor + HHstatin_ + ethnic_B + ethnic_D + ethnic_F + ethnic_G + ethnic_H + ethnic_I + ethnic_J + ethnic_N + ethnic_O + ethnic_R + ethnic_S + ethnic_U + kid16_17_U + eqpdays

f1_nn
#create the neural network. Architecture - two hidden layers with 10 and 3 neurons respectively. 
nn <- neuralnet(f1_nn,data=train_cln,hidden=c(10, 3), threshold = 0.6)


# Visual plot of the model
plot(nn)

churn_test_prob = neuralnet::compute(nn,test_cln)

churn_test_prob_result<-churn_test_prob$net.result

churn_nn_pred = prediction(churn_test_prob_result, test_cln$churn)
perf_nn_churn <- performance(churn_nn_pred, "tpr", "fpr") 
plot(perf_nn_churn)

auc_ROCR_nn <- performance(churn_nn_pred, measure = "auc")
auc_ROCR_nn
auc_ROCR_nn@y.values[[1]]

######################## End: Neutal network with  relevent fields ################################################################################


