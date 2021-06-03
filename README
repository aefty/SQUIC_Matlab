### Installation

1) Download SQUIC_Release_Source from https://github.com/aefty/SQUIC_Release_Source,
   (follow given installation instructions)

2) Download SQUIC_MATLAB from https://github.com/aefty/SQUIC_Matlab 

3) In the terminal go to folder where SQUIC_MATLAB is installed, open compile_mex file 
   and check that 

  ```
  MEX='path/to/matlab/bin/mex'     # e.g. '/Applications/MATLAB_R2021a.app/bin/mex' (check Matlab version)
  LIBSQUIC_DIR='path/toLibSQUIC    # e.g. '/Users/user_name'  
  ```

4) in SQUIC_MATLAB folder run
    
   ```angular2
   sh compile_mex
   ```
   
   if successful, you should now have a SQUIC_MATLAB.mex* file. 


### Example

```
p=1000;
n=100;
lambda=.3;
Y = randn(p,n);

[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda);

% optionally change input for max_iter, inv_tol, etc. 
max_iter = 40;
[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda, max_iter);
```

All parameters, default settings and description of return values can be found under
`help SQUIC`. 







