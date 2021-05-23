data = dlmread('swissroll.dat');
X = data(1:50,:);

X1 = X - mean(X)

C = cov(X1)
[evec,eval] = eig(C);


k=3
function [W,m] = pca_learning(X,k)
m = mean(X,2);
[W,~] = eigs(cov(X'), k);