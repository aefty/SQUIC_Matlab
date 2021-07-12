## SQUIC for Python
### Sparse Quadratic Inverse Covariance Estimation
This is the SQUIC algorithm, made available as a Python package. 
SQUIC tackles the statistical problem of estimating large sparse 
inverse covariance matrices. This estimation poses an ubiquitous 
problem that arises in many applications e.g. coming from the 
fields mathematical finance, geology and health. 
SQUIC belongs to the class of second-order L1-regularized 
Gaussian maximum likelihood methods and is especially suitable 
for high-dimensional datasets with limited number of samples. 
For further details please see the listed references.

### References

[1] Eftekhari A, Pasadakis D, Bollhöfer M, Scheidegger S, Schenk O (2021). Block-Enhanced PrecisionMatrix Estimation for Large-Scale Datasets.Journal of Computational Science, p. 101389.

[2] Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.

[3] Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation on high-performance computing architectures. In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.

### Installation

1) Download SQUIC_Release_Source from https://github.com/aefty/SQUIC_Release_Source
   (follow provided installation instructions).

2) Download SQUIC_MATLAB from https://github.com/aefty/SQUIC_Matlab 

3) In the terminal go to folder where SQUIC_MATLAB is installed, open compile_mex file 
   and check that 

```
MEX='path/to/matlab/bin/mex'     # e.g. '/Applications/MATLAB_R2021a.app/bin/mex' (check Matlab version)
LIBSQUIC_DIR='path/toLibSQUIC    # e.g. '/Users/user_name'  
```

4) in SQUIC_MATLAB folder run
    
```
sh compile_mex
```
   
if successful, you should now have a SQUIC_MATLAB.mex* file. 


### Example

```
p=1024;
n=100;
lambda=.4;
Y = randn(p,n);

% to compute sample covariance matrix
[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC_S(Y, lambda);

% to compute sparse sampel covariance matrix X, and its inverse W, etc.
[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda, max_iter);
```

All parameters, default settings and description of return values can be found under
`help SQUIC`. 







