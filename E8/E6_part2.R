setwd('/home/tuomas/R/Projects/DATA.STAT.770/E8/Chronic_Kidney_Disease')
data = read.csv('chronic_kidney_disease_full.arff', head=T, skip=145)
data = data.frame(data)
labels = data[,25]
# Clean the data
real = c(1,2,10,11,12,13,14,15,16,17,18)
data = data[,real]
data[] = lapply(data, as.numeric)
mask = rowSums(is.na(data))==0
data = data[mask,]
labels = labels[mask]
# Normalize features
sds = lapply(data, sd)
data_train = data/sds
data_train = scale(data_train)
data_train = matrix(data_train, ncol=11, byrow = F)

library(kohonen)
# blood pressure = column 2 (X80)
# red blood cell count = column 11 (X5.2)
data_train2 = cbind(data_train[,2],data_train[,11]) 
colnames(data_train2) <- c('blood pressure', 'red blood cell count')
somnet1 = som(data_train2, somgrid(5,8,'hexagonal'))
plot(somnet1, shape=c('straight'), main='Attribute variation among prototypes')

somnet2 = xyf(data_train, factor(labels), somgrid(5,8,'hexagonal'))
plot(somnet2, shape=c('straight'), main='Majority class of data for each prototype')


################################################################
somnet = som(data_train, somgrid(5,8,'hexagonal'))
plot(somnet, type='property', property=somnet$data[,2],cex=0.5)






################################################################
somnet3 = som(data_train, somgrid(5,8,'hexagonal'))
som3.pred = predict(somnet3, newdata = data_train)
plot(som3.pred$predictions)
#######################33
data(wines)
set.seed(7)

training <- sample(nrow(wines), 150)
Xtraining <- scale(wines[training, ])
