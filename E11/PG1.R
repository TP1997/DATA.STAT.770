library(snedata)
library(dimRed)
library("scatterplot3d")
library(scales)
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E11')

# Generate swissroll data to obtain also color values
data.swiss = swiss_roll(n = 1000)
swiss.color = data.swiss$color
data.swiss = data.swiss[,1:3]
# Plot original
par(mfrow=c(1,2))
scatterplot3d(data.swiss$x,data.swiss$y,data.swiss$z,
              color=swiss.color, pch=16, angle=22*pi)
title('Swissroll original')
# tSNE
emb.swiss = embed(data.swiss, "tSNE", perplexity = 80)
emb.swiss = as.data.frame(emb.swiss@data)
# Plot embedded
plot(emb.swiss$tSNE1,emb.swiss$tSNE2,col=swiss.color,pch=16)
title('Swissroll embedded')

data.hs = read.table('halfsphere.dat')
r = cos(data.hs$V1)
r = rescale(r,to=c(0,255))
g = cos(data.hs$V2+120)
g = rescale(g,to=c(0,255))
b = cos(data.hs$V3-120)
b = rescale(b,to=c(0,255))
hs.color = rgb(r,g,b,maxColorValue=255)
# Plot original
scatterplot3d(data.hs, angle=32*pi, color=hs.color, pch=16)
title('Halfsphere original')
# tSNE
emb.hs = embed(data.hs, "tSNE", perplexity = 80)
emb.hs = as.data.frame(emb.hs@data)
# Plot embedded
plot(emb.hs$tSNE1,emb.hs$tSNE2,col=hs.color,pch=16)
title('Halfsphere embedded')



data.iris = iris[,1:4]
iris.color = rep("#FF0000", sum(iris$Species=="setosa"))
iris.color = append(iris.color, rep("#00FF00", sum(iris$Species=="versicolor")))
iris.color = append(iris.color, rep("#0000FF", sum(iris$Species=="virginica")))
# Plot original
scatterplot3d(data.iris[,1:3], angle=92*pi, color=iris.color, pch=16)
title('Iris original\n(First 3 dimensions used)')
# tSNE
emb.iris = embed(data.iris, "tSNE", perplexity = 30)
emb.iris = as.data.frame(emb.iris@data)
# Plot embedded
plot(emb.iris$tSNE1,emb.iris$tSNE2,col=iris.color,pch=16)
title('Iris embedded')
