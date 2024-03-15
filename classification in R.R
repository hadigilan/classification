#
#	Note:
#	This script utilize the examples of James et al. (2017)
#	An Introduction to Statistical Learning with Applications in R
#
#



#
#	packages
#

#install.packages("ISLR", dependencies=T)
#install.packages("class", dependencies=T)
#install.packages("ROCR", dependencies=T)
#install.packages("pROC", dependencies=T)



library (MASS)				 # for lda
library(ISLR)				 # for dataset
library(class)				 # for knn
#library(ROCR)
library(pROC)





#
#	the dataset
#


data(Smarket)


# Smarket is a dataset containing percentage returns for the S&P 500 stock index, 250 days, from the beginning of 2001 until the end of 2005.
# Lag1-Lag5: the percentage returns for each of the five previous trading days 
# Volume: the number of shares traded on the previous day, in billions
# Today: the percentage return on the date in question
# Direction: whether the market was Up or Down on today's date


head(Smarket)
attach( Smarket )
names(Smarket )
dim(Smarket )
summary (Smarket )

#
# According to the market efficieny hypothesis, previous stock returns shouldn't be correlated with the today's return. 
#

R<- cor(Smarket [,-c(1, 9)])						# correlation matrix

#
#	covariance matrix of inputs can be obtained by the correlation matrix by S= Ds * R * Ds
#

Ds<- diag(apply(Smarket[,-c(1, 9)], 2, sd ))				
COV<- Ds %*% R %*% Ds							# Covariance matrix

apply(Smarket[,-c(1, 9)], 2, var )					# Just for checking


#
#	splitting data
#


train<- Smarket[Year !=2005 , ]
table(train$Year)
test<- Smarket[Year==2005,]
table(test$Year)
dim(test)


#
#	Logistic regression
#

#logis<- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume , data=train , family = binomial )
logis<- glm(Direction ~ Lag1 + Lag2 , data=train , family = binomial )

summary ( logis)
coef(logis)


#
# 	We know that value 1 corresponds to the probability of the market going up, rather than down, because Direction is a factor
#	and levels function indicates that the first level (the base one) is Down.
#

levels(Direction)

#
# 	predicted probabilities, i.e., P(Y = 1|X)
#

logis.prob<- predict (logis, type = "response", newdata=test)
head(logis.prob)

#
#	confusion matrix for logistic
#

logis.label<- ifelse(logis.prob>= 0.5, "UP", "DOWN")
#data.frame(logis.prob=logis.prob, logis.label=logis.label)

table(actual=test$Direction, prediction=logis.label)
#(35 + 106) / 252										 #	correct classification rate

#
#	ROC of logistic
#

par(mfrow=c(1,2))

logistic.roc <- roc(test$Direction, logis.prob )
plot(logistic.roc )
auc(logistic.roc )



#
#	Linear discriminant analysis (LDA)
#


#lda.fit<- lda(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume  + Today, prior=c(0.5, 0.5) ,data=train)
lda.fit<- lda(Direction ~ Lag1 + Lag2 , prior=c(0.5, 0.5), data=train)
lda.fit$scaling


#
#	prediction using LDA
#

lda.pred<- predict(lda.fit, newdata=test)
names(lda.pred)
lda.pred$posterior
#head(cbind(class=lda.pred$class, prob.down=lda.pred$posterior[,1], prob.up=lda.pred$posterior[,2]))

lda.prob<- lda.pred$posterior[, 2]						# class UP is the main class which we are predicting
lda.label<- lda.pred$class
#cbind(lda.prob, lda.label)



#
#	confusion matrix for LDA
#


table(actual=test$Direction, prediction=lda.label)		
#mean(lda.label == test$Direction)			
# (64+74) /252									# correct classification rate

#
#	ROC of LDA
#

#par(mfrow=c(1,2))
#pred.lda <- prediction(lda.prob, lda.label)								# prediction is a built-in function from ROCR package to generate required data for ROC curve
#slotNames(pred )
#roc.lda<- performance(prediction.obj=pred.lda , measure="tpr", x.measure="fpr")
#plot(roc.lda)
#abline(a=0, b= 1)													# the 45-degree line, which represents, on average, the performance of a random classifier



lda.roc <- roc(test$Direction, lda.prob)
plot(lda.roc )
auc(lda.roc )




#
#	Quadratic discriminant analysis (QDA)
#
	
#qda.fit<- qda(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume  + Today, prior=c(0.5, 0.5) , data=train)
qda.fit<- qda(Direction ~ Lag1 + Lag2 , prior=c(0.5, 0.5) , data=train)
qda.fit$scaling



#
#	prediction with qda	
#


qda.pred<- predict(qda.fit, newdata=test)
qda.prob<- qda.pred$posterior[, 2]						# class UP is the main class which we are predicting
qda.label<- qda.pred$class


#
#	confusion for QDA
#


table(test$Direction, qda.label)
# (55 + 83) / 252


#
#	ROC of QDA
#


qda.roc <- roc(test$Direction, qda.prob)
plot(qda.roc )
auc(qda.roc )


#	
#	KNN method
#

knn.pred1<- knn(train[,c("Lag1" ,"Lag2")], test[,c("Lag1" ,"Lag2")], k = 1, cl=train$Direction, prob=TRUE)
knn.pred3<- knn(train[,c("Lag1" ,"Lag2")], test[,c("Lag1" ,"Lag2")], k = 3, cl=train$Direction, prob=TRUE)

knn.prob<- attributes(knn.pred3)$prob

#
#	confusion matrix for KNN
#

table(test$Direction, knn.pred1)
# (43 + 83) / 252



table(test$Direction, knn.pred3)
# (48 + 86) / 252



#
#	ROC of KNN
#


knn.roc <- roc(test$Direction, knn.prob)
plot(knn.roc )
auc(knn.roc )



