rm(list=ls())
setwd('E:\\AVdatafest\\Loan Prediction')
train<-read.csv('train.csv')
test<-read.csv('test.csv')
Loan_ID<-test$Loan_ID
summary(train)
summary(test)
library(DMwR)
train[train==""]<-NA
test[test==""]<-NA
train=train[,-c(1,2)]
test=test[,-c(1,2)]
library(mice)
trainmice<-mice(train)
testmice<-mice(test)
train1<-complete(trainmice)
test1<-complete(testmice)
table(train$Gender,train$Loan_Status)

plot(train1$ApplicantIncome)
#preprocess######
summary(train1)
test1$Loan_Status<-NA
full<-rbind(train1,test1)
plot(full$CoapplicantIncome)
summary(full)
full$full<-full$ApplicantIncome+full$CoapplicantIncome
full$full=ifelse(full$full>10000,10000,full$full)
full$ApplicantIncome<-ifelse(full$ApplicantIncome>7500,7500, full$ApplicantIncome)
full$full<-log(full$full)
full$CoapplicantIncome<-ifelse(full$CoapplicantIncome>4500,4500,full$CoapplicantIncome)
plot(full$CoapplicantIncome)
library(infotheo)
x<-discretize(full$ApplicantIncome,disc='equalwidth', nbins=20)
y<-discretize(full$CoapplicantIncome,disc='equalwidth',nbins=20)
z<-discretize(full$full,disc='equalwidth',nbins=20)
full$ApplicantIncome<-x$X
full$CoapplicantIncome<-y$X
full$full<-z$X
train1<-full[1:nrow(train),]
library(infotheo)


test1<-full[-(1:nrow(train)),]
summary(train1)
plot(train1$ApplicantIncome,train1$Loan_Status)
library(ggplot2)
ggplot(train1,aes(ApplicantIncome,Loan_Status))+geom_jitter()
table(train1$ApplicantIncome)
#glmnet#######
library(glmnet)
# logistic regression#####
set.seed(100)

glm1<-glm(Loan_Status~.,train1,family=binomial(link = logit))
Loan_Status<-predict(glm1,test1,type='response')
Loan_Status<-ifelse(Loan_Status>.58,'Y','N')
table(Loan_Status)
Sample_Submission<-data.frame(Loan_ID,Loan_Status)
write.csv(Sample_Submission,'logistic.csv')
plot(glm1)
library(MASS)
glm2<-stepAIC(glm1)
anova(glm1,glm2, test ="Chisq")

library(lmtest)
lrtest(glm1,glm2)

#ROC curve
library(ROCR)
p<-predict(glm1,train1[,-14],type='response')
pr<-prediction(p,train1$Loan_Status)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
library(pROC)
library(pROC)

my_roc <- roc(train1$Loan_Status,p)
coords(my_roc, "best", ret = "threshold")
library(MASS)
glm2<-stepAIC(glm1)

p<-predict(glm2,train1,type='response')
pr<-prediction(p,train1$Loan_Status)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
glm2

#ann models#####
library(nnet)
anmodel<-nnet(Loan_Status~.,train1[,-c(1,2,4,5)],size=4,skip=T,trace=T,MaxNWts=5000)
anpred<-predict(anmodel,test1[,-c(1,2,4,5)])
anpred<-ifelse(anpred>0.55,'Y','N')
table(anpred)
Loan_Status<-anpred
neural_pred<-data.frame(Loan_ID,Loan_Status) 
write.csv(neural_pred,'neural.csv')
summary(train)
summary(test)
#random forest####

levels(train1)<-c('NA','NO','YES')
library(randomForest)
model<-randomForest(Loan_Status~.,train1[,])
forestpred<-predict(model,test1[,])
Sample_Submission<-data.frame(Loan_ID,Loan_Status)
write.csv(Sample_Submission,'random.csv')
plot(importance(model))
plot(importance(model))


#SVM for binary prediction####
library(kernlab)
sv<-ksvm(Loan_Status~.,train1, C = 0.1)
svpred<-predict(sv, test1)
table(svpred)
Sample_Submission$Loan_Status<-svpred
write.csv(Sample_Submission,'ksvm.csv')

#fast Adaboost####
library(fastAdaboost)
ada<-adaboost(Loan_Status~.,train1[,-c(1,2,4,5)],nIter = 10)
adapred<-predict(ada,test1[,-c(1,2,4,5)])
adapred
Sample_Submission$Loan_Status<-adapred$class
write.csv(Sample_Submission,'adaboost.csv')

#ensembleing####
#logistic+randomforest+aaboost+svm

testSet$pred_majority<-as.factor(ifelse(testSet$pred_rf=='Y' & 
        testSet$pred_knn=='Y','Y',
       ifelse(testSet$pred_rf=='Y' & testSet$pred_lr=='Y','Y',
        ifelse(testSet$pred_knn=='Y' & testSet$pred_lr=='Y','Y','N'))))


#look
majorityNO<-ifelse(adapred$class=='Y'& svpred=='Y','Y',ifelse(Loan_Status=='N'& forestpred=='N','N','Y'))
ymajor<-ifelse(Loan_Status=='Y'& majorityNO=='N','N','Y')
Sample_Submission$Loan_Status<-majorityNO
write.csv(Sample_Submission,'ensemble2.csv')
sparse_matrix <- sparse.model.matrix(Loan_Status ~ .-1, data = train1)
output_vector = train1[,12] == "Y"
labels = train1['Loan_Status']
df_train = train1[-grep('Loan_Status', colnames(train1))]
data<-data.frame()




Loan_Status<-ifelse(forestpred=='N'&Loan_Status=='N','N','Y')
Sample_Submission$Loan_Status<-Loan_Status
write.csv(Sample_Submission,'annlog.csv')
table(anpred)


#decision tree C50######
library(C50)
levels(train1$Gender)[1] = "missing"
levels(train1$Married)[1] = "missing"
levels(train1$Dependents)[1] = "missing"
levels(train1$Self_Employed)[1] = "missing"
dt<-C5.0(train1[,-12],train1[,12])
dtpredict<-predict(dt,test1)
dtpredict
table(dtpredict)
Sample_Submission$Loan_Status<-dtpredict
write.csv(Sample_Submission,'decisiontree.csv')
