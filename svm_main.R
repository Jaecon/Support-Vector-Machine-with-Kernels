rm(list=ls())

## Load packages
library(glmnet)
library(MASS)  # Package needed to generate correlated precictors
library(readr)
library(tidyr) # use to drop NAs
library(e1071)
library(kernlab) # ksvm
library(svmpath) # svmpath
library(ggplot2)
library(gmodels) # CrossTable
library(BBmisc) # Normalize data
library(miceadds) # load.Rdata
library(klaR) # NavieBayes
library(caret) # createFolds()
library(doParallel)
registerDoParallel(cores = 7)
#################################################################
## read in the data
#################################################################

airline <- read.csv("twostage.csv", colClasses = c(rep('factor',15),rep('numeric',22),'factor'))

dta <- subset(airline, select=-c(oligo_route_r,r)) 

colnames(dta)

dta.testSet.id <- which(dta$dominant_route_r==1)

set.seed(123)
testSet.id <- split(sample(dta.testSet.id), 1:10) 
testSet.id
# 10-fold dominant routes


#############################
## nth trial with parallel ##
#############################

list.rad.tunes <- list()
mat.test.errors <- vector()

for(i in 2:10){
  test.set <- dta[testSet.id[[i]],]
  train.set <- dta[-testSet.id[[i]],]
  
  X.test <- model.matrix(entry_within4~.-1, test.set)
  Y.test <- test.set$entry_within4
  X.train <- model.matrix(entry_within4~.-1, train.set)
  Y.train <- train.set$entry_within4
  
  #colnames(X.test)
  p1 <- X.test[,c(1:17)]
  p2 <- BBmisc::normalize(X.test[,-c(1:17)], method = "standardize", margin = 2, on.constant = "stop")
  X.test <- cbind(p1,p2)
  #colnames(X.test)
  #head(X.test[,1:17])
  #summary(X.test[,-c(1:17)])
  
  p1 <- X.train[,c(1:17)]
  p2 <- BBmisc::normalize(X.train[,-c(1:17)], method = "standardize", margin = 2, on.constant = "stop")
  X.train <- cbind(p1,p2)
  
  weight <- unname( table(Y.train)[1]/table(Y.train)[2] )
  
  set.seed(4444)
  rad.tune <- e1071::tune(e1071::svm, X.train, Y.train
                     ,ranges = list(cost = 2^(-10:10),gamma = 10^(-3:3))
                     ,class.weights = c("0"= 1,"1" = weight)
                     ,kernel = "radial"
  )
  mat.test.errors[i] <- sum(predict(rad.tune$best.model, X.test) != Y.test)/length(Y.test)
  list.rad.tunes[[i]] <- rad.tune
  save(rad.tune, file = paste0('rad-',i,'.RData'))
  print(paste0(i," Done"))
}

#err.rad <- sum(predict(rad.tune$best.model, X.test) != Y.test)/length(Y.test)
#best.parmeters <- rad.tune$best.parameters


res <- data.frame("cost"=double(),"gamma"=double(),"best_performance"=double(), "test_errors"=double())
for (i in 1:10 ){
  res[i,]<-cbind(list.rad.tunes[[i]]$best.parameters, list.rad.tunes[[i]]$best.performance, mat.test.errors[i])
}

#######################
## Test-error cruves ##
#######################

test.set <- dta[dta.testSet.id,]
train.set <- dta[-dta.testSet.id,]

X.test <- model.matrix(entry_within4~.-1, test.set)
Y.test <- test.set$entry_within4
X.train <- model.matrix(entry_within4~.-1, train.set)
Y.train <- train.set$entry_within4

p1 <- X.test[,c(1:17)]
p2 <- BBmisc::normalize(X.test[,-c(1:17)], method = "standardize", margin = 2, on.constant = "stop")
X.test <- cbind(p1,p2)
#colnames(X.test)
#head(X.test[,1:17])
#summary(X.test[,-c(1:17)])

p1 <- X.train[,c(1:17)]
p2 <- BBmisc::normalize(X.train[,-c(1:17)], method = "standardize", margin = 2, on.constant = "stop")
X.train <- cbind(p1,p2)

weight <- unname( table(Y.train)[1]/table(Y.train)[2] )

K <- 10
NBC.test.error <- rep(-99, K) # K number of test error 
for (k in 1:K){
  NBC.model <- klaR::NaiveBayes(X.train, Y.train, usekernel=T)
  yv.hat <- predict(NBC.model, X.test, threshold = 1e-4 )
  
  NBC.test.error[k] <- sum(Y.test != yv.hat$class)/length(Y.test)
}

NBC.CV.score <- mean(NBC.test.error)
NBC.CV.score



#### Probability
# cost = 2^(-10:10),gamma = 10^(-3:3)
tot.X <- model.matrix(entry_within4 ~ . -1, dta)
tot.Y <- dta$entry_within4
colnames(tot.X)
p1 <- tot.X[,c(1:17)]
p2 <- BBmisc::normalize(tot.X[,-c(1:17)], method = "standardize", margin = 2, on.constant = "stop")
tot.X <- cbind(p1,p2)

weight <- unname( table(tot.Y)[1]/table(tot.Y)[2] )

dt.tot.performance <- data.frame(cost = numeric(), gamma = double(), error = double())

for (g in 10^(-3:3)){
  res.per.g <- foreach(c=2^(-10:10), .combine = rbind) %dopar%{
    cv <- e1071::svm(tot.X, tot.Y,
                     scale=F,
                     class.weights = c("0"=1,"1"=weight),
                     kernel = 'radial',
                     cost = c, gamma = g
                     )
    err <- sum(cv$fitted!=tot.Y)/length(tot.Y)
    data.frame(cost = cv$cost, gamma = cv$gamma, error = err)
  }
  dt.tot.performance <- rbind(dt.tot.performance, res.per.g)
}

with(dt.tot.performance, dt.tot.performance[which(error==0), ])

# cost = 32, gamma = 0.1
best.model <- svm(tot.X, tot.Y,
                  scale=F,
                  class.weights = c("0"=1,"1"=weight),
                  kernel = 'radial',
                  cost = 32, gamma = 0.1,
                  probability = T
)
tot.test.X <- tot.X[dta.testSet.id,]
pred_response <- predict(best.model, tot.test.X, probability=T)
head(pred_response) # bullshit prob= 0 or 1



#--------------------------------------------------------------------
tc <- seq(1e-01,1e+01,0.1)
length(tc)
gmm <- c(0.001,0.01, 0.1, 1)

TE.radial <- list()

for (g in gmm) {
  te <- foreach(i = tc, .combine = rbind) %dopar% {
    cv<- e1071::svm(X.test, Y.test
                    ,scale = F
                    ,class.weights = c("0"= 1,"1" = 15.77)
                    ,kernel = 'radial'
                    ,cost = i
                    ,gamma = g)
    errr <- sum(cv$fitted!=Y.test)/length(Y.test)
    data.frame(cost = cv$cost, test.error = errr)
  }
  TE.radial <- c( TE.radial, list(te) )
}
names(TE.radial) <- gmm

windows()
par(mfrow=c(1,length(gmm)-1))
plot(TE.radial[[1]], main = expression(paste(gamma," = 0.001")), xlab='C', ylab='Test Error')
plot(TE.radial[[2]], main = expression(paste(gamma," = 0.01")), xlab='C', ylab='Test Error')
plot(TE.radial[[3]], main = expression(paste(gamma," = 0.1")), xlab='C', ylab='Test Error')


