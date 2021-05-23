setwd('/home/tuomas/R/Projects/DATA.STAT.770/E8')
data = read.table("swissroll.dat")

library(dimRed)




dat <- loadDataSet("3D S Curve")
leim <- LaplacianEigenmaps()
emb <- leim@fun(dat, leim@stdpars)


plot(emb@data@data)
