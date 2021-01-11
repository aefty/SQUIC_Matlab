rng(10)
p=10;
n=30000;

iC=sprandsym(p,.3,1./(1:p));
full(iC)
C=inv(iC);
mu=zeros(p,1);

data_train = mvnrnd(mu,C,n)';

lambda=.01;
max_iter=10;
drop_tol=1e-2;
term_tol=1e-10;
M=speye(p,p)*eps;
 
X0=speye(p);
W0=speye(p);
data_test=1;
verbose=1;

[X,W,info]=SQUIC(data_train, lambda,max_iter,drop_tol,term_tol,verbose);

full(X)