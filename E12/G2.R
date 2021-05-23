library("scatterplot3d")
library(scales)
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E12')

plottopng = function(root, title, col){
  setwd(root)
  nerv.files = list.files('.')
  for (i in 1:11) {
    data = read.table(nerv.files[i])
    lambda = (i-1)*0.1
    plotname = paste0(strsplit(nerv.files[i],".dat")[[1]][1], '.png')
    png(plotname)
    plot(data$V1, data$V2, col=col, pch=16)
    title(paste(title,lambda))
    dev.off()
  }
}

# Swissroll data
data.swiss = read.table('swissroll.dat')
col.swiss = read.table('swissroll_col.dat')$V1

png('swissroll_orig.png')
scatterplot3d(data.swiss$V1, data.swiss$V2, data.swiss$V3,
              color=col.swiss, pch=16, angle=27*pi, scale.y=2.5)
title('Swissroll original')
dev.off()

plottopng('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/swissroll',
          'Swissroll NeRV, lambda =',
          col.swiss)

# halfsphere data
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E12')
data.hs = read.table('halfsphere.dat')
col.hs = read.table('halfsphere_col.dat')$V1

png('halfsphere_orig.png')
scatterplot3d(data.hs$V1, data.hs$V2, data.hs$V3,
              color=col.hs, pch=16, angle=27*pi, scale.y=0.5)
title('Halfsphere original')
dev.off()

plottopng('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/tst',
          'Halfsphere NeRV, lambda =X',
          col.hs)

# CKD data
setwd('/home/tuomas/R/Projects/DATA.STAT.770/E12')
data.ckd = read.table('ckd.dat')
col.ckd = read.table('ckd_labels.dat')
col.ckd[col.ckd=='ckd'] = "#FF0000"
col.ckd[col.ckd=='notckd'] = "#00FF00"

plottopng('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/ckd',
          'CKD NeRV, lambda =',
          col.ckd$V1)

nerv.files = list.files('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/ckd')
plotname = paste0(strsplit(nerv.files[1],".dat")[[1]][1], '.png')

setwd('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/tst')
nerv.files = list.files('/home/tuomas/R/Projects/DATA.STAT.770/E12/nervs/tst')
data = read.table(nerv.files[2])
plot(data$V1, data$V2, col=col.hs, pch=16)
