library(snedata)
library(dimRed)
library("scatterplot3d")
library(scales)
root = '/home/tuomas/R/Projects/DATA.STAT.770/E12'
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E12')

# Generate swissroll data to obtain also color values
data.swiss = swiss_roll(n = 1000)
swiss.color = data.swiss$color
data.swiss = data.swiss[,1:3]

write.table(data.swiss, file=paste(root,'swissroll.dat',sep='/'),
            row.names=F, col.names=F)
write.table(swiss.color, file=paste(root,'swissroll_col.dat',sep='/'),
            row.names=F, col.names=F)

# Generate halfsphere colors
data.hs = read.table('halfsphere.dat')
r = cos(data.hs$V1)
r = rescale(r,to=c(0,255))
g = cos(data.hs$V2+120)
g = rescale(g,to=c(0,255))
b = cos(data.hs$V3-120)
b = rescale(b,to=c(0,255))
hs.color = rgb(r,g,b,maxColorValue=255)

write.table(hs.color, file=paste(root,'halfsphere_col.dat',sep='/'),
            row.names=F, col.names=F)

# Generate cleaned CKD and colors
data = read.csv('chronic_kidney_disease_full.arff', head=T, skip=145)
data.cdk = data.frame(data)
labels = data.cdk[,25]

# Clear ckd/t and no samples
mask = as.logical((labels=='ckd')+(labels=='notckd'))
data.cdk = data.cdk[mask,]
labels = labels[mask]

# Clean the data
real = c(1,2,10,11,12,13,14,15,16,17,18)
data.cdk = data.cdk[,real]
data.cdk[] = lapply(data.cdk, as.numeric)
mask = rowSums(is.na(data.cdk))==0
data.cdk = data.cdk[mask,]
labels = labels[mask]

write.table(data.cdk, file=paste(root,'ckd.dat',sep='/'),
            row.names=F, col.names=F)
write.table(labels, file=paste(root,'ckd_labels.dat',sep='/'),
            row.names=F, col.names=F)
