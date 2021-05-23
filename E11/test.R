angle_hs = function(mat, c){
  res = numeric()
  nc = norm(c)
  for (i in 1:dim(mat)[2]) {
    m=t(as.numeric(mat[,i]))
    nm=norm(m)
    res[i] =(m%*%t(c))/(nm*nc)
  }
  return(res)
}

data.hs = read.table("halfsphere.dat", sep=" ", dec=".", head=FALSE)[,-1]
angle.x = angle_hs(t(as.matrix(data.hs[,1:2])), cbind(0,1))
angle.y = angle_hs(t(as.matrix(data.hs[,2:3])), cbind(0,1))
angle.z = angle_hs(t(as.matrix(data.hs[,-2])), cbind(0,1))