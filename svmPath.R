rm(list=ls())

## Load packages
library(MASS)  # Package needed to generate correlated precictors
library(readr)
#library(tidyr) # use to drop NAs
library(e1071)
library(kernlab) # ksvm
library(ggplot2)
library(gmodels) # CrossTable
library(BBmisc) # Normalize data
library(miceadds) # load.Rdata
library(doParallel)
registerDoParallel(cores = 7)
#################################################################
## read in the data
#################################################################

airline <- read.csv("twostage.csv", colClasses = c(rep('factor',19),rep('numeric',23)))

dta <- subset(airline, select=-c(dominant_route_r,oligo_route_r,legacy_route_r,AA_post,r)) # 1 factor level in AA_post
colnames(dta)


p1 <- dta[,c(1:15)]
p2 <- normalize(dta[,-c(1:15)], method = "standardize", margin = 2, on.constant = "stop")
sdta <- cbind(p1,p2)
attach(sdta)

##################################### svm
plot.svmpath1 <- function(x1,x2){
  
  ssdta <- data.frame(entry_within4, x1, x2)
  
  lin.fit <- svm(entry_within4 ~ x1 + x2, data =ssdta
                 ,kernel = "linear"
                 ,class.weights = c("0"= 1,"1" = 11.40))

  rad.fit <- svm(entry_within4 ~ x1 + x2, data =ssdta
                 ,kernel = "radial"
                 ,class.weights = c("0"= 1,"1" = 11.40))
  
  plot(lin.fit, ssdta)
  plot(rad.fit, ssdta)
}


##################################### ksvm
plot.svmpath2 <- function(x1,x2){
  svp.lin <- ksvm(entry_within4 ~ x1 + x2, data = sdta
                  ,type="C-svc"
                  ,kernel="vanilladot"
                  ,class.weights = c("0"= 1,"1" = 11.40)
  )
  
  svp.rad <- ksvm(entry_within4 ~ x1 + x2, data = sdta
                  ,type="C-svc"
                  ,kernel="rbfdot"
                  ,class.weights = c("0"= 1,"1" = 11.40)
  )
  
  plot(svp.lin, data=sdta)
  plot(svp.rad, data=sdta)
}

plot.svmpath1(rt_distance, hhi_direct_connect)
plot.svmpath2(rt_distance, hhi_direct_connect)

plot.svmpath1(Pre_WNpresence_A, gmean_population)
plot.svmpath2(Pre_WNpresence_A, gmean_population)

detach(sdta)

