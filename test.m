p=1000;
n=100;
lambda=.3;
Y = randn(p,n);

[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda,0);

verbose = 0;


%[S,info_times_S]=SQUIC_S(Y, lambda);
