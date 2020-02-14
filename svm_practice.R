rm(list=ls())

## Load packages
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
library(doParallel)
registerDoParallel(cores = 7)
#################################################################
## read in the data
#################################################################

airline <- read.csv("twostage.csv", colClasses = c(rep('factor',19),rep('numeric',23)))

dta <- subset(airline, select=-c(dominant_route_r,oligo_route_r,legacy_route_r,AA_post,r)) # 1 factor level in AA_post
colnames(dta)

with(dta,CrossTable(dta[,"t_0"]>83,dta[,"entry_within4"]))

test.index <- which(dta[,"t_0"]>83) # ~2013 Q4: training

test.set <- dta[test.index,]
train.set <- dta[-test.index,]

X.test <- model.matrix(entry_within4~.-1, test.set)
Y.test <- test.set$entry_within4
X.train <- model.matrix(entry_within4~.-1, train.set)
Y.train <- train.set$entry_within4

colnames(X.test)
p1 <- X.test[,c(1:18)]
p2 <- normalize(X.test[,-c(1:18)], method = "standardize", margin = 2, on.constant = "stop")
X.test <- cbind(p1,p2)
colnames(X.test)
head(X.test[,1:18])

p1 <- X.train[,c(1:18)]
p2 <- normalize(X.train[,-c(1:18)], method = "standardize", margin = 2, on.constant = "stop")
X.train <- cbind(p1,p2)

#################################################################
#  SVM
#################################################################

#obj <- tune(svm, Species~., data = i2, 
#            ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
#            class.weights= c("setosa" = 1, "versicolor" = 10),
#            tunecontrol = tune.control(sampling = "cross")
#           )
#

#table(Y.train)[1]/table(Y.train)
## for class.weights w.r.t. Y=1

weight <- unname( table(Y.train)[1]/table(Y.train)[2] )

set.seed(4444)
lin.tune <- tune(svm, X.train, Y.train
                 ,ranges = list(cost = c(2^(-10:3),10^(1:2)))
                 ,class.weights = c("0"= 1,"1" = weight)
                 ,kernel = "linear"
)
lin.tune


set.seed(4444)
rad.tune <- tune(svm, X.train, Y.train
                 ,ranges = list(cost = 2^(-10:10),gamma = 10^(-3:3))
                 ,class.weights = c("0"= 1,"1" = weight)
                 ,kernel = "radial"
)
rad.tune


save(lin.tune, file = 'lin.RData')
save(rad.tune, file = 'rad.RData')







######################################################################
# Results
######################################################################

#################
## Error rates ##
#################


#temp <- load.Rdata2(filename = 'temp.RData')

err.lin <- sum(predict(lin.tune$best.model, X.test) != Y.test)/length(Y.test)
err.rad <- sum(predict(rad.tune$best.model, X.test) != Y.test)/length(Y.test)

error.rates <- c(err.lin, err.rad)
names(error.rates) <- c("linear","radial")
error.rates

best.parmeters <- list(linear = lin.tune$best.parameters
                       ,radial = rad.tune$best.parameters
                       )
best.parmeters

#######################
## Test-error cruves ##
#######################

#table(Y.test)[1]/table(Y.test)

# RADIAL - best.model
# cost = 10 // gamma = 0.1


#library(stats) # density(data$sbp,kernel = "epanechnikov")
validation.id <- split(sample(1:dim(X.test)[1]),1:10) # 10-fold CV for NBC test error

K <- 10
NBC.test.error <- rep(-99, K) # K number of test error 
for (k in 1:K){
  xv <- X.test[validation.id[[k]],]
  xt <- X.test[-validation.id[[k]],]
  yv <- Y.test[validation.id[[k]]]
  yt <- Y.test[-validation.id[[k]]]
  
  NBC.model <- klaR::NaiveBayes(xt, yt, usekernel=T)
  yv.hat <- predict(NBC.model, xv, threshold = 1e-4 )
  
  NBC.test.error[k] <- sum(yv!=yv.hat$class)/length(yv)
}

NBC.CV.score <- mean(NBC.test.error)
NBC.CV.score


tc <- seq(1e-01,1e+01,0.1)
length(tc)
gmm <- c(0.001,0.01, 0.1, 1)

TE.radial <- list()

weight <- unname( table(Y.train)[1]/table(Y.train)[2] )

for (g in gmm) {
  te <- foreach(i = tc, .combine = rbind) %dopar% {
    cv<- e1071::svm(X.test, Y.test
                      ,scale = F
                      ,class.weights = c("0"= 1,"1" = weight)
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

