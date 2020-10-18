# Prostate Cancer Data Analysis.


# We don't want to many digits showing
options(digits=2)



suppressMessages(library(tidyverse)) # Used for piping %>%

prostate_data=read.table(file='prosdats.txt',header=FALSE) # Read the data in.

names(prostate_data)=c('id','lcavol',	'lweight',	'age',	'lbph',	'svi',	'lcp',	'gleason',	'pgg45','lpsa',	'train') # Read in the names manually


prostate_data=prostate_data%>%  # We don't need to consider ID variable so we can drop it.
  dplyr::select(-(id)) 


prostate_data_train=prostate_data%>%  # We only wish to keep the training data to model.
  filter(train==TRUE)

prostate_data_test=prostate_data%>%  # We only wish to keep the training data to model.
  filter(train==FALSE)


# Reorder the variables for better visualization. The following pairs plot should look similar to the pairs plot in ESL page 3.

prostate_data_train=prostate_data_train[ ,c('lpsa','lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45')]

predictors_train=prostate_data_train[ ,c('lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45')]

predictors_test=prostate_data_test[ ,c('lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45')]


dim(prostate_data_train)
dim(prostate_data_test)

pairs(prostate_data_train,col="darkorchid3")
cor(prostate_data_train)

predictors_scaled=as.data.frame(scale(predictors_train))

prostate_data_train=data.frame(prostate_data_train$lpsa,predictors_scaled)

names(prostate_data_train)= c('lpsa', 'lcavol', 'lweight',    'age',   'lbph'  , 'svi' ,   'lcp', 'gleason',  'pgg45')


predictors_scaled_test=as.data.frame(scale(predictors_test))

prostate_data_test=data.frame(prostate_data_test$lpsa,predictors_scaled_test)

names(prostate_data_test)= c('lpsa', 'lcavol', 'lweight',    'age',   'lbph'  , 'svi' ,   'lcp', 'gleason',  'pgg45')




## Linear

prostate_linear=lm(lpsa~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45,data=prostate_data_train)
summary(prostate_linear)


predictions=predict(prostate_linear,newdata=data.frame(prostate_data_train))

RSS1=sum((predictions-prostate_data_train$lpsa)^2)

prostate_linear_reduced=lm(lpsa~lcavol+lweight+lbph+svi,data=prostate_data_train)

summary(prostate_linear_reduced)

predictions_reduced=predict(prostate_linear_reduced,newdata=data.frame(prostate_data_train))

RSS0=sum((predictions_reduced-prostate_data_train$lpsa)^2)


predict_test=predict(prostate_linear,newdata=data.frame(prostate_data_test))

RSS_test=sum((predict_test-prostate_data_test$lpsa)^2)


## Moving on from least squares

library(caret)
library(leaps)

best_subsets <- regsubsets(lpsa~., data = prostate_data_train, nvmax = 8,nbest=1)
summary(best_subsets)

linear_best_subset=lm(lpsa~lcavol+lweight,data=prostate_data_train)
summary(linear_best_subset)


#We need to calculate RSS for the intercept only model first. RSS_int

prostate_intercept=lm(lpsa~1,data=prostate_data_train)


intercept_pred=predict(prostate_intercept,new.data=prostate_data_train)
RSS_intercept=sum((intercept_pred-prostate_data_train$lpsa)^2)

plot(x=0:8,c(RSS_intercept,summary(best_subsets)$rss),xlim=c(0,8),ylim=c(0,100),xlab="Subset Size k",ylab="Residual Sum-of-Squares",col='red')
lines(x=0:8,c(RSS_intercept,summary(best_subsets)$rss),col='red')




suppressMessages(library(glmnet))

#Choose an intermediate value of Lambda , say 0.5

model_crossv <- glmnet(x=as.matrix(predictors_scaled), as.matrix(prostate_data_train$lpsa), alpha = 0, lambda = 0.5, standardize = TRUE)

coef(model_crossv)



model_lasso <- glmnet(x=as.matrix(predictors_scaled), as.matrix(prostate_data_train$lpsa), alpha = 1, lambda = 0.21, standardize = TRUE)

coef(model_lasso)

#intermediate lambda value, Notice some predictors are actually discarded when using lasso. Here I just chose an intermediate value to get a similar result as in the ESL.




