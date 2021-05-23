library("tuneR");
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E3')
original<-readWave("the_entertainer.wav");
originalvector<-as.matrix(original@left);
nelements<-dim(originalvector)[1]*dim(originalvector)[2];
blocksize<-100;
nblocks<-floor(nelements/blocksize);
featuredata<-matrix(0,nblocks,blocksize);
for (blockindex in 1:nblocks)
{
  featuredata[blockindex,]<-t(originalvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]);
}

# -------------------your code begins here----------------

# Insert your code here to compute a PCA reconstruction of the feature data into 
# a new matrix called reconstructed_featuredata

components = 15
cm = cov(featuredata)
evv = eigen(cm)
W = evv$vectors[,1:components]

z = t(W) %*% t(featuredata)

reconstructed_featuredata = W %*% z
reconstructed_featuredata = t(reconstructed_featuredata)

# -------------------your code ends here----------------


reconstructedvector<-matrix(0,nblocks*blocksize,1);
for (blockindex in 1:nblocks)
{
  reconstructedvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]<-t(reconstructed_featuredata[blockindex,]);
}

reconstructedaudio<-original;
reconstructedaudio@left<-as.vector(reconstructedvector);

writeWave(reconstructedaudio,"pca_musicreconstruction.wav")

# Plotting
par(mfrow=c(2,1))
seconds = 5
plot(reconstructedaudio@left[1:(seconds*44100)], type='l', main='Reconstructed',
     xlab='', ylab='')
plot(original@left[1:(seconds*44100)], type='l', main='Original',
     xlab='', ylab='')
