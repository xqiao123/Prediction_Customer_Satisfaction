library(cluster)
library(Rtsne)
library(ggplot2)
library(dplyr)
library(clustrd)
library(fossil)
library(MASS)
library(class)
library(hmeasure)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(party)
library(hmeasure)


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
data=data.frame(air$Age, air$Flight.Distance, air$Departure.Delay.in.Minutes, air$Arrival.Delay.in.Minutes, gender, cusType, travelType, class, seat, depart, food, gate, wifi, entertain, online, booking, board, legRoom, baggage, check, clean, onlineBoard, response)
str(data)

#Scatter Plot
air2=data.frame(air$medi, air$Age, air$Flight.Distance, air$Arrival.Delay.in.Minutes)
colnames(air2)=c('medi','age','flight.distance','arrival.delay.in.minutes')
air2$medi=ifelse(air2$medi=='satisfied',1,0)
air2
plot(air2$age, air2$medi)
plot(air2$flight.distance, air2$medi)
plot(air2$arrival.delay.in.minutes, air2$medi)

#--Tidying things up a bit--#
colnames(data)<-c('age','flight.distance','departure.delay.in.minutes','arrival.delay.in.Minutes','gender', 'cusType','traveltype','class','seat','depart','food','gate','wifi','entertain','online','booking','board','legRoom','baggage','check','clean','onlineBoard','response')
predictors=data.frame(air$Age,air$Flight.Distance, air$Departure.Delay.in.Minutes, air$Arrival.Delay.in.Minutes, gender, cusType, travelType, class, seat, depart, food, gate, wifi, entertain, online, booking, board, legRoom, baggage, check, clean, onlineBoard)
response.df=data.frame(response)

#Gower's dist#
transformed.air=daisy(data,metric = "gower")

#PAM#
pam_air=pam(transformed.air, diss = TRUE, k = 4)
pam_air$clusinfo
gowerPAMclustering=pam_air$clustering
data[pam_air$medoids, ]

#--So I'm wondering what could be my best dimension+cluster combination---#
best.combination=tuneclus(data, nclusrange = 3:7, ndimrange = 2:5,
                          method = c("clusCA","iFCB","MCAk"),
                          criterion = "asw", dst = "full", alpha = 1, alphak = 1,
                          center = TRUE, scale = TRUE, rotation = "none", nstart = 100,
                          smartStart = NULL, seed = NULL)

best.combination #5 cluesters + 4-dimension

#--5 clusters & 4 dimensions is the best clusCA&MCAk combination--##
#no plot for 5-cluster 
mix54=clusmca(data, nclus=5, ndim=4, method=c("clusCA","MCAk"),
alphak = .5, nstart = 100, smartStart = NULL, gamma = TRUE,
inboot = FALSE, seed = NULL)

mix44=clusmca(data, nclus=4, ndim=4, method=c("clusCA","MCAk"),
alphak = .5, nstart = 100, smartStart = NULL, gamma = TRUE,
inboot = FALSE, seed = NULL)

mix55=clusmca(data, nclus=5, ndim=5, method=c("clusCA","MCAk"),
alphak = .5, nstart = 100, smartStart = NULL, gamma = TRUE,
inboot = FALSE, seed = NULL)

mix54.clustering=mix54$cluster
mix44.clustering=mix44$cluster
mix55.clustering=mix55$cluster

Cluster.Assignment=data.frame(as.vector(gowerPAMclustering),as.vector(mix54.clustering),as.vector(mix44.clustering),as.vector(mix55.clustering))
colnames(Cluster.Assignment)<-c("Gower+PAM","MIX54","MIX44","MIX55")
rownames(Cluster.Assignment)<-as.factor(1:nrow(air))

#--library(fossil)---#
#rand.index(Cluster.Assignment[,1],Cluster.Assignment[,6])
similarity.matrix=matrix(0,ncol(Cluster.Assignment),ncol(Cluster.Assignment))
for(i in 1:ncol(Cluster.Assignment))
{
for(j in 1:ncol(Cluster.Assignment))
{
similarity.matrix[i,j]=rand.index(Cluster.Assignment[,i],Cluster.Assignment[,j])
}
}
rownames(similarity.matrix)=colnames(Cluster.Assignment)
colnames(similarity.matrix)=colnames(Cluster.Assignment)
plot(hclust(as.dist(1-similarity.matrix),method = "complete"))

#--Rand index--#
rand.index(gowerPAMclustering,as.vector(mix54.clustering)) #0.8
rand.index(gowerPAMclustering,as.vector(mix44.clustering)) #0.8
rand.index(as.vector(mix54.clustering),as.vector(mix44.clustering)) #0.9
rand.index(as.vector(mix55.clustering),as.vector(mix44.clustering)) #0.9
