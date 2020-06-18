#Load the necessary libraries
library(dplyr)
library(stringr)
library(tibble)
library(readr)
library(tidyr)
library(caret)
library(data.table)
library(rpart)
library(gam)
library(nnet)
library(lars)
library(kernlab)

#To automatically install packages if needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lars)) install.packages("lars", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(fastICA)) install.packages("fastICA", repos = "http://cran.us.r-project.org")
if(!require(pls)) install.packages("pls", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")

####GATHERING, ARRANGING, AND CLEANING####
#Download and rearrange red wine data
url1<-"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dl1<-tempfile()
download.file(url1,dl1)
red<-read.csv(dl1, sep=";")

#Download and rearrange white wine data
url2<-"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
dl2<-tempfile()
download.file(url2,dl2)
white<-read.csv(dl2, sep=";")

#Add a column characterising the type of wine
red<- red%>%mutate(type = "red")
white<- white%>%mutate(type = "white")

#Combine the wines
wine<-rbind(red,white)

#change type to a factor
wine$type<-as.factor(wine$type)

#Check for NA's and remove if they exist
for (x in 1:12){print(sum(is.na(wine[,x]))) }
wine<-na.omit(wine)

#remove unnecessary data
rm(url1,url2,dl1,dl2,x)

####EXPLORATION AND VISUALIZATION####
#Check the structure of the data
str(wine)

#Visualize the the difference in quality between red and white
histogram(red$quality, xlab="Red Wine Quality")
histogram(white$quality, xlab="White Wine Quality")
summary(red$quality)
summary(white$quality)

#Compare averages for different variables for each wine
var<-data.frame(Variables=colnames(wine[,1:12]), RED=c(1:12), WHITE=c(1:12))
for(x in 1:12){var[x,2]<-mean(red[,x])
  var[x,3]<-mean(white[,x])}
var%>%knitr::kable() #This explains why the wine color had to be added as a factor

#Correlation between variables and quality
cors<-data.frame(Variables=colnames(wine[1:12]), cor_coeff= (1:12))
for(x in 1:12){cors[x,2]<-(cor(wine[,x],wine[,12]))}
cors%>%knitr::kable() 

#Visualize quality vs variables
for (x in 1:11){(plot(wine[,12],wine[,x], xlab= colnames(wine[12]), 
                      ylab=colnames(wine[x])))}

####FURTHER CLEANING###
#Remove outlier
outlier<-boxplot(wine$quality,plot=FALSE)$out
wine_updated<-wine%>%filter(!(quality %in% outlier))

####MODELING####
#Introducing RMSE function
RMSE<-function(true,pred){
  sqrt(mean((true-pred)^2))
}

#Partition data only 20% in test set
set.seed(1, sample.kind = "Rounding")
test_index<-createDataPartition(wine_updated$quality, times=1,p=0.2,list=FALSE)
train_set<-wine_updated[-test_index,]
test_set<-wine_updated[test_index,]

#Linear Regression 
lm<-train(quality~.,data=train_set, method= "lm")
y_lm<-predict(lm, test_set)

#Create table to store RMSE
results<-data.frame(Method="Linear Regression", RMSE=RMSE(test_set$quality,y_lm))

#Least Angle Regression 
lars<-train(quality~.,data=train_set, method= "lars", tuneGrid=data.frame(fraction=seq(0,1,0.2)))
plot(lars)
y_lars<-predict(lars, test_set)
results<-rbind(results,data.frame(Method="Least Angle Regression", 
                                  RMSE=RMSE(test_set$quality,y_lars)))

#Independent Component Regression 
icr<-train(quality~.,data=train_set, method= "icr", tuneGrid=data.frame(n.comp=seq(1,10,1)))
plot(icr)
y_icr<-predict(icr, test_set)
results<-rbind(results,data.frame(Method="Independent Component Regression",
                                  RMSE=RMSE(test_set$quality,y_icr)))

#Principal Component Analysis 
pcr<-train(quality~.,data=train_set, method= "pcr", tuneGrid=data.frame(ncomp=seq(1,13,1)))
plot(pcr)
y_pcr<-predict(pcr, test_set)
results<-rbind(results,data.frame(Method="Principal Component Analysis", 
                                  RMSE=RMSE(test_set$quality,y_pcr)))

#CART 
rpart<-train(quality~.,data=train_set, method= "rpart", tuneGrid= data.frame(cp=seq(0,0.05,len=25)))
plot(rpart)
y_rpart<-predict(rpart, test_set)
results<-rbind(results,data.frame(Method="CART",RMSE=RMSE(test_set$quality,y_rpart)))
#To visualize the final tree
plot(rpart$finalModel)
text(rpart$finalModel, cex=0.75)

#Bagged CART 
treebag<-train(quality~.,data=train_set, method= "treebag")
y_treebag<-predict(treebag, test_set)
results<-rbind(results,data.frame(Method="Bagged CART", RMSE=RMSE(test_set$quality,y_treebag)))

#Stochastic Gradient Boosting 
control<-trainControl(method="cv", number= 10, p=0.9)
grid<-expand.grid(interaction.depth=seq(5,10), n.trees=seq(100,700,100),
                  shrinkage=0.1, n.minobsinnode=10)
gbm<-train(quality~.,data=train_set, method= "gbm", tuneGrid=grid, trControl=control)
plot(gbm)
y_gbm<-predict(gbm, test_set)
results<-rbind(results,data.frame(Method="Stochastic Gradient Boosting",
                                  RMSE=RMSE(test_set$quality,y_gbm)))

#Generalized Additive Model using LOESS 
grid<- expand.grid(span=seq(0.15,0.65,len=10), degree=1)
gamLeoss<-train(quality~.,data=train_set, method= "gamLoess", tuneGrid= grid)
y_gamLeoss<-predict(gamLeoss, test_set)
results<-rbind(results,data.frame(Method="Generalized Additive Model using LOESS",
                                  RMSE=RMSE(test_set$quality,y_gamLeoss)))

#Support Vector Machines with Linear Kernel 
control<-trainControl(method="cv", number= 10, p=0.9)
svmL<-train(quality~.,data=train_set, method= "svmLinear",trControl=control) 
y_svmL<-predict(svmL, test_set)
svmO<-data.frame(Method="SVM with Linear Kernel", RMSE=RMSE(test_set$quality,y_svmL))

#Support Vector Machines with Polynomial Kernel 
control<-trainControl(method="cv", number= 10, p=0.9)
svmP<-train(quality~.,data=train_set, method= "svmPoly",trControl=control) 
y_svmP<-predict(svmP, test_set)
svmO<-rbind(svmO,data.frame(Method="SVM with Polynomial Kernel",
                            RMSE=RMSE(test_set$quality,y_svmP)))

#Support Vector Machines with Radial Basis Function Kernel 
control<-trainControl(method="cv", number= 10, p=0.9)
svmR<-train(quality~.,data=train_set, method= "svmRadial",trControl=control) 
y_svmR<-predict(svmR, test_set)
svmO<-rbind(svmO,data.frame(Method="SVM with Radial Basis Function Kernel",
                            RMSE=RMSE(test_set$quality,y_svmR)))

#Support Vector Machines with Radial Basis Function Kernel tuning C
grid<-expand.grid(C=seq(1,3,0.5),sigma=0.086)
svmR2<-train(quality~.,data=train_set, method= "svmRadial", tuneGrid= grid,trControl=control) 
plot(svmR2)
y_svmR2<-predict(svmR2, test_set)
svmO<-rbind(svmO,data.frame(Method="SVM with Radial Basis Function Kernel (tuning C (2.5))",
                            RMSE=RMSE(test_set$quality,y_svmR2)))

#Support Vector Machines with Radial Basis Function Kernel tuning C and Sigma
grid<-expand.grid(C=seq(2,3,0.25),sigma=seq(0,0.1,0.02))
svmR3<-train(quality~.,data=train_set, method= "svmRadial", tuneGrid= grid, trControl=control) 
plot(svmR3)
y_svmR3<-predict(svmR3, test_set)
svmO<-rbind(svmO,data.frame(Method="SVM with Radial Basis Function Kernel (tuning C (2.75) 
                            and sigma (0.1))", RMSE=RMSE(test_set$quality,y_svmR3)))

#Support Vector Machines with Radial Basis Function Kernel tuning Sigma 
grid<-expand.grid(C=2.75,sigma=seq(0.08,0.16,0.02))
svmR4<-train(quality~.,data=train_set, method= "svmRadial", tuneGrid= grid, trControl=control) 
y_svmR4<-predict(svmR4, test_set)
plot(svmR4)
svmO<-rbind(svmO,data.frame(Method="SVM with Radial Basis Function Kernel (C (2.75)
                            tuning sigma (0.12))", RMSE=RMSE(test_set$quality,y_svmR4)))
results<-rbind(results,data.frame(Method="Support Vector Machines with Radial Basis Function Kernel",
                                  RMSE=RMSE(test_set$quality,y_svmR4)))


#Inspect the trained model from each method to find the optimal RMSE value
lm
lars
icr
pcr
rpart
treebag
gbm
gamLeoss
svmR4

#Creat a table with train set RMSE
train_r<-results
train_r[1,2]<-lm$results["RMSE"]
train_r[2,2]<-lars$results["RMSE"][3,]
train_r[3,2]<-icr$results["RMSE"][7,]
train_r[4,2]<-pcr$results["RMSE"][12,]
train_r[5,2]<-rpart$results["RMSE"][3,]
train_r[6,2]<-treebag$results["RMSE"]
train_r[7,2]<-gbm$results["RMSE"][41,]
train_r[8,2]<-gamLeoss$results["RMSE"][10,]
train_r[9,2]<-svmR4$results["RMSE"][3,]
train_r%>%arrange(desc(RMSE))%>%knitr::kable()

#Ensemble for the two with the lowest RMSE for the train set 
y_ensemble<-(y_gbm+y_svmR4)/2
results<-rbind(results,data.frame(Method="Ensemble:GBM and SvmRadial",
                                  RMSE=RMSE(test_set$quality,y_ensemble)))

#Visualize the RMSE tables for results and the SVM methods in descending order
svmO%>%arrange(desc(RMSE))%>%knitr::kable()
results%>%arrange(desc(RMSE))%>%knitr::kable()

#Visualize the best method vs prediction
t<-test_set[,12:13]
t<-cbind(t,y_ensemble)
t%>%ggplot(aes(quality, round(y_ensemble,1), color=type))+geom_count(alpha= 0.3)
                +ylab("Predicted quality")
