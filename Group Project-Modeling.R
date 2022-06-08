library(cluster)
library(Rtsne)
library(ggplot2)
library(dplyr)
library(MASS)
library(class)
library(hmeasure)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(party)
library(earth)
library(caretEnsemble)

##Preparation
air=airline

#In Classification model: keep response and predictors same data type eg. factor 
response=as.factor(air$medi)

gender=as.factor(air$Gender)
cusType=as.factor(air$Customer.Type)
travelType=as.factor(air$Type.of.Travel)
class=as.factor(air$Class)

seat=as.factor(air$Seat.comfort)
depart=as.factor(air$Departure.Arrival.time.convenient)
food=as.factor(air$Food.and.drink)
gate=as.factor(air$Gate.location)
wifi=as.factor(air$Inflight.wifi.service)
entertain=as.factor(air$Inflight.entertainment)
online=as.factor(air$Online.support)
booking=as.factor(air$Ease.of.Online.booking)
board=as.factor(air$On.board.service)
legRoom=as.factor(air$Leg.room.service)
baggage=as.factor(air$Baggage.handling)
check=as.factor(air$Checkin.service)
clean=as.factor(air$Cleanliness)
onlineBoard=as.factor(air$Online.boarding)

#--Let's bring in all of them, even the categorical ones----#
data=data.frame(air$Age, air$Flight.Distance, air$Arrival.Delay.in.Minutes, gender, cusType, travelType, class, seat, depart, food, gate, wifi, entertain, online, booking, board, legRoom, baggage, check, clean, onlineBoard, response)
str(data)

#--Tidying things up a bit--#
colnames(data)<-c('age','flight.distance','arrival.delay.in.Minutes','gender', 'cusType','traveltype','class','seat','depart','food','gate','wifi','entertain','online','booking','board','legRoom','baggage','check','clean','onlineBoard','response')
predictors=data.frame(air$Age,air$Flight.Distance, air$Arrival.Delay.in.Minutes, gender, cusType, travelType, class, seat, depart, food, gate, wifi, entertain, online, booking, board, legRoom, baggage, check, clean, onlineBoard)
response.df=data.frame(response)

##########
tr=createDataPartition(data$response, p=3/4, list = F)

tr.pred=predictors[tr,]
test.pred=predictors[-tr,]
tr.response=response.df[tr,]
test.response=response.df[-tr,]

#---Some data transformation (scaling, centering, etc.), notice how the categoricals are untouched---#
trans=preProcess(tr.pred, method = c('knnImpute','center','scale'))
trans.tr.pred=predict(trans,tr.pred)
trans.test.pred=predict(trans,test.pred)

#---Notice how "createDataPartition" maintains balance:--#
prop.table(table(response))
prop.table(table(tr.response))
prop.table(table(test.response))

#---Now a 10-fold cross validated logistic---#
ctrl=trainControl(method = "repeatedcv", number = 10, repeats=5, savePredictions = T)

cv.logistic=train(x=trans.tr.pred,y=tr.response,method='glm',trControl=ctrl,family='binomial')
cv.logistic
system.time(train(x=trans.tr.pred,y=tr.response,method='glm',trControl=ctrl,family='binomial'))
pred=predict(cv.logistic,newdata = trans.test.pred)
pred
pred.prob=predict(cv.logistic,newdata = trans.test.pred, type = "prob")
pred.prob
confusionMatrix(data=pred, test.response, positive = "satisfied")
summary(cv.logistic$finalModel)
cv.logistic$results

varImp(cv.logistic)
plot(varImp(cv.logistic))


#################
#--Now, one pruned tree--#
#################
cv.tree=train(x=trans.tr.pred,y=tr.response,method='rpart',trControl=ctrl)
cv.tree
plot(cv.tree)
pred.tree=predict(cv.tree,newdata = trans.test.pred)
pred.tree

head(predict(cv.tree,newdata = trans.test.pred, type = "prob"))
confusionMatrix(data=pred.tree,test.response,positive = "satisfied")
cv.tree$results
system.time(train(x=trans.tr.pred,y=tr.response,method='rpart',trControl=ctrl))

varImp(cv.tree)
plot(varImp(cv.tree))

# advanced Tree plot
cont=rpart.control(minsplit=3, cp = 0.02423168,
                   maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
                   surrogatestyle = 0, maxdepth = 30)

modelTree2<- rpart(response~., data = data, method = 'class', control = cont)
rpart.plot(modelTree2, extra = "auto")
modelTree2


###################
#--Next, a bagged tree---#
###################
cv.bagged.tree=train(x=trans.tr.pred,y=tr.response,method='treebag',trControl=ctrl)
cv.bagged.tree
#plot(cv.bagged.tree)
pred.bagged.tree=predict(cv.bagged.tree,newdata = trans.test.pred)
pred.bagged.tree

head(predict(cv.bagged.tree,newdata = trans.test.pred, type = "prob"))
cv.bagged.tree$results
confusionMatrix(data=pred.bagged.tree,test.response,positive = "satisfied")
system.time(train(x=trans.tr.pred,y=tr.response,method='treebag',trControl=ctrl))

varImp(cv.bagged.tree)
plot(varImp(cv.bagged.tree))


###################
#--Next, a random forest---#
###################
cv.randomforest=train(x=trans.tr.pred,y=tr.response,method='cforest',trControl=ctrl)
cv.randomforest
plot(cv.randomforest)
pred.randomforest=predict(cv.randomforest,newdata = trans.test.pred)
pred.randomforest
cv.randomforest$results
confusionMatrix(data=pred.randomforest,test.response,positive = "satisfied")
system.time(train(x=trans.tr.pred,y=tr.response,method='cforest',trControl=ctrl))

varImp(cv.randomforest)
plot(varImp(cv.randomforest))


###################
#--Moving on to boosting---#
###################
cv.boosted=train(x=trans.tr.pred,y=tr.response,method='gbm',trControl=ctrl, metric='Accuracy') #--The method changes to 'gbm'#
cv.boosted
plot(cv.boosted)
pred.boost=predict(cv.boosted,newdata = trans.test.pred)
pred.boost
cv.boosted$results
confusionMatrix(data=pred.boost,test.response,positive = "satisfied")
system.time(train(x=trans.tr.pred,y=tr.response,method='gbm',trControl=ctrl))
varImp(cv.boosted)
plot(varImp(cv.boosted))




