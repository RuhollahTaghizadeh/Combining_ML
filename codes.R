# library
library(caret)
library(tidyverse)
library(dplyr)
library(purrr)
library(Metrics)
library(parallelMap)
library(PerformanceAnalytics)
library(mlr)
library(e1071)
library(ranger)
library(xgboost)
library(Cubist)
library(randomForest)
library(earth)
library(DescTools)

# import data
data = read.csv("LUCAS.csv")
data = data[complete.cases(data),]
names(data)
xy_LUCAS <- data[, c(198,199)]
target_Lucas <- data[,c(186:197)]
cov_Lucas <- data[,c(1:185)]

# pre-processing of covariates
dim(cov_Lucas)
nzv <- nearZeroVar(cov_Lucas)
cov_Lucas <- cov_Lucas[, -nzv]
dim(cov_Lucas)

highlyCor <- findCorrelation(cov_Lucas, cutoff = .98)
cov_Lucas <- cov_Lucas[,-highlyCor]
dim(cov_Lucas)

scale <- preProcess(cov_Lucas, method = c("center", "scale"))
cov_Lucas <- predict(scale, newdata = cov_Lucas)
dim(cov_Lucas)
str(cov_Lucas)

# tuning the hyper-parameters of ML models
df1 <- cbind(cov_Lucas, sand=target_Lucas$sand)

# hyper-parameters tuning
# start
parallelStartSocket(4)

# pre-defined parameters
ctrl = makeTuneControlRandom(maxit = 30L)
rdesc = makeResampleDesc("CV", iters = 3L)
tsk = makeRegrTask(data = df1, target = "sand")

# knn 
getParamSet("regr.fnn")
discrete_knn = makeParamSet(
  makeDiscreteParam("k", values = c(1, 2, 3, 4, 5, 6, 10, 15, 20, 25)))
res_knn = tuneParams("regr.fnn", task = tsk, resampling = rdesc,
                 par.set = discrete_knn, control = ctrl)

# Rf 
getParamSet("regr.ranger")
discrete_rf = makeParamSet(
  makeIntegerParam("num.trees", lower = 1L, upper = 1000L),
  makeIntegerParam("mtry", lower = 1L, upper = 50L),
  makeDiscreteParam("min.node.size", values = c(1, 2, 4, 5,7, 10, 15)))
res_rf = tuneParams("regr.ranger", task = tsk, resampling = rdesc,
                     par.set = discrete_rf, control = ctrl)

# XGB 
getParamSet("regr.xgboost")
discrete_xgb = makeParamSet(
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  makeNumericParam("eta", lower = .1, upper = .5),
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x))
res_xgb = tuneParams("regr.xgboost", task = tsk, resampling = rdesc,
                    par.set = discrete_xgb, control = ctrl)

# cubist 
getParamSet("regr.cubist")
discrete_cub = makeParamSet(
  makeIntegerParam("committees", lower = 1L, upper = 100L),
  makeIntegerParam("rules", lower = 1L, upper = 100L),
  makeNumericParam("extrapolation", lower = 0, upper = 100),
  makeIntegerParam("neighbors", lower = 0L, upper = 9L))
res_cub = tuneParams("regr.cubist", task = tsk, resampling = rdesc,
                     par.set = discrete_cub, control = ctrl)


# stop
parallelStop()


# cross validation of base learners
nval=nrow(df1) 
kfold=10  
j <- sample.int(nval)
df1 <-df1 [j,] 
folds <- cut(seq(1,nval),breaks=kfold,labels=FALSE) 
act_pre_knn = list()  
act_pre_rf = list()  
act_pre_xgb = list()  
act_pre_cub = list()  


for (i in 1:kfold) {
  idx   <- which(folds==i) 
  model_knn <- train.kknn(sand~., data = df1[-idx,], kmax = 15)
  pred_knn  <- predict (model_knn, newdata=df1[idx,]) 
  act_pre_knn[[i]] <- data.frame(A = df1$sand[idx] , P = pred_knn)}

for (i in 1:kfold) {
  idx   <- which(folds==i) 
  model_rf <- ranger(formula = sand~., data = df1[-idx,], num.trees = 510, mtry = 21, min.node.size=2)
  pred_rf  <- predict (model_rf, df1[idx,]) 
  act_pre_rf[[i]] <- data.frame(A = df1$sand[idx] , P = pred_rf$predictions)}

cov_xgb <- as.matrix(df1[,-ncol(df1)])
tar_xgb <- as.matrix(df1[,ncol(df1)])

for (i in 1:kfold) {
  idx   <- which(folds==i) 
  model_xgb <- xgboost(data = cov_xgb[-idx,], label = tar_xgb[-c(idx)], nrounds=275, max_depth=4, eta=0.145, lambda=0.129)
  pred_xgb  <- predict (model_xgb, cov_xgb[idx,]) 
  act_pre_xgb[[i]] <- data.frame(A = tar_xgb[idx] , P = pred_xgb)}

for (i in 1:kfold) {
  idx   <- which(folds==i) 
  model_cub <- cubist(x = cov_xgb[-idx,], y = tar_xgb[-c(idx)],committees=46, rules=67, extrapolation=28, neighbors=0)
  pred_cub <- predict (model_cub, cov_xgb[idx,]) 
  act_pre_cub[[i]] <- data.frame(A = tar_xgb[idx] , P = pred_cub)}


# create data frame based on the predicted values
dfpredicts = list()
for (i in 1: kfold) { 
  dfpredicts [[i]] = data.frame (m1=act_pre_knn[[i]]$P,m2=act_pre_rf[[i]]$P,
                                 m3=act_pre_xgb [[i]]$P,m4=act_pre_cub [[i]]$P,
                                 A=act_pre_cub [[i]]$A)}

# define formula for super learning
a=names(dfpredicts [[1]][1:4])
b = paste("A","~")
formula= as.formula (paste(b,paste(a,collapse="+")))

df_sup = rbind(dfpredicts[[1]],dfpredicts[[2]],dfpredicts[[3]],
               dfpredicts[[4]],dfpredicts[[5]],dfpredicts[[6]],
               dfpredicts[[7]],dfpredicts[[8]],dfpredicts[[9]],
               dfpredicts[[10]])

folds_sup <- cut(seq(1,nrow(df_sup)),breaks=kfold,labels=FALSE) 


act_pre_sup <- list()
for (i in 1:kfold) {
  idx   <- which(folds_sup==i) 
  model_sup <- svm(formula, data=df_sup[-idx,])
  pred_sup <- predict (model_sup, df_sup[idx,]) 
  act_pre_sup[[i]] <- data.frame(A = df_sup$A[idx] , P = pred_sup)}

act_pre_sup_df <- rbind(act_pre_sup[[1]],act_pre_sup[[2]],act_pre_sup[[3]],
                        act_pre_sup[[4]],act_pre_sup[[5]],act_pre_sup[[6]],
                        act_pre_sup[[7]],act_pre_sup[[8]],act_pre_sup[[9]],
                        act_pre_sup[[10]])

# combine all of them
act_pre_all <- cbind(df_sup[,-ncol(df_sup)], sup = act_pre_sup_df$P, act=act_pre_sup_df$A)
names(act_pre_all) <- c("kNN", "RF", "XGB", "Cubist", "SuperLearner", "Actual")
cor(act_pre_all)
pairs(act_pre_all)
summary(act_pre_all)

RMSE_result <- list()
for (i in 1:5) {
RMSE_result[[i]] <- sqrt(mean((act_pre_all$Actual - act_pre_all[[i]])^2))}
RMSE_result <- data.frame(RMSE_result)
names(RMSE_result) <- c("kNN", "RF", "XGB", "Cubist", "SuperLearner")

nRMSE_result <- list()
for (i in 1:5) {
tmp <- sqrt(mean((act_pre_all$Actual - act_pre_all[[i]])^2))
nRMSE_result[[i]] <- tmp/mean(act_pre_all$Actual)}
nRMSE_result <- data.frame(nRMSE_result)
names(nRMSE_result) <- c("kNN", "RF", "XGB", "Cubist", "SuperLearner")

corr_result <- list()
for (i in 1:5) {
corr_result[[i]] <- cor(act_pre_all$Actual , act_pre_all[[i]])}
corr_result <- data.frame(corr_result)
names(corr_result) <- c("kNN", "RF", "XGB", "Cubist", "SuperLearner")

LCC_result <- list()
for (i in 1:5) {
tmp <- CCC(act_pre_all$Actual,act_pre_all[[i]])
LCC_result[[i]] <- tmp$rho.c[[1]] }
LCC_result <- data.frame(LCC_result)
names(LCC_result) <- c("kNN", "RF", "XGB", "Cubist", "SuperLearner")


error_index <- rbind(RMSE_result, nRMSE_result, corr_result, LCC_result)
rownames(error_index) <- c("RMSE", "normalized_RMSE","correlation", "concordance")


# plot
act_pre_all_plot <- act_pre_all%>%
  select(-Actual) %>%
  gather(model, predicted)%>%
  add_column(actual = rep(act_pre_all$Actual, 5))


P_01 <- act_pre_all_plot %>% 
  ggplot(aes(actual, predicted)) + 
  geom_point(alpha = 0.50) +
  geom_abline(slope = 1, intercept = 0, color = "red") + 
  facet_wrap(~ model) + 
  theme_minimal() + 
  labs(x = "actual",y = "predicted")

# export results
write.csv(act_pre_all, "act_pre_all_150922.csv")
write.csv(error_index, "error_index_150922.csv")
tiff("P_01_150922.tiff", units="in", width=5, height=5, res=300)
P_01
dev.off()

