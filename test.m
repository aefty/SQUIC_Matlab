p=1000;
n=100;
lambda=.3;
Y = randn(p,n);
[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda);