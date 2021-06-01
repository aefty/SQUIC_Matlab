function [X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC(Y, lambda, max_iter, inv_tol, term_tol,verbose, M, X0, W0)
    % SQUIC : Sparse Inverse Covariance Estimation
    % [X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC(Y, lambda, max_iter, inv_tol, term_tol,verbose, M, X0, W0)
    %
    % input arguments : 
    %
    % Y:                Data matrix p x p consisting of p>2 random variables and n>1 samples.
    % l:                Sparsity parameter as a nonzero positive scalar.
    % max_iter:         [default 100] Maximum number of Newton iterations as a nonnegative integer
    % inv_tol:          [default 1e-3] Termination tolerance as a nonzero positive scalar.
    % term_tol:         [default 1e-3] Dropout tolerance as a nonzero positive scalar.
    % verbose:          [default 1] Verbosity level as 0 or 1.
    % M:                [default None] Sparsity structure matrix as a sparse p x p matrix.
    % X0:               [default None] Initial value of precision matrix as a sparse p x p matrix.
    % W0:               [default None] Initial value of inverse precision matrix as a sparse p x p  matrix.
    %
    % return values : 
    %
    % X:                Precision matrix as a sparse p x p matrix.
    % W:                Inverse precision matrix as a sparse p x p matrix.
    % info_times:       List of different compute times.
    % info_objective:   List of objective function values at each Newton iteration.
    % info_logdetX:     Log-determinant of the final precision matrix.
    % info_trSX:        Trace of sample covariance matrix times the final precision matrix.
    %
    %
    % Example
    %
    % 
    %
    % See also SQUIC_S.
    
    setenv('KMP_DUPLICATE_LIB_OK','TRUE');

    [p,n]= size(Y);
    
    if(nargin == 2)
        max_iter =  100;
        inv_tol =  1e-3;
        term_tol =  1e-3;
        verbose =  1;        
        M=sparse(p,p);
        W0=speye(p,p);
        X0=speye(p,p);      
    end

    if(nargin == 3)
        inv_tol =  1e-3;
        term_tol =  1e-3;
        verbose =  1;        
        M=sparse(p,p);
        W0=speye(p,p);
        X0=speye(p,p);      
    end
       
    if(nargin == 4)
        term_tol =  1e-3;
        verbose =  1;        
        M=sparse(p,p);
        W0=speye(p,p);
        X0=speye(p,p);      
    end
       
    if(nargin == 5)
        verbose =  1;        
        M=sparse(p,p);
        W0=speye(p,p);
        X0=speye(p,p);      
    end

    if(nargin == 6)    
        M=sparse(p,p);
        W0=speye(p,p);
        X0=speye(p,p);      
    end

    if(nargin == 7 || nargin == 8)    
        W0=speye(p,p);
        X0=speye(p,p);      
    end    

    if(p<3)
        error('#SQUIC: number of random variables (p) must larger than 2');
    end
    if(n<2)
        error('#SQUIC: number of samples (n) must be larger than 1 .');
    end
    if(lambda<=0)
        error('#SQUIC: lambda must be great than zero.');
    end
    if(max_iter<0)
        error('#SQUIC: max_iter cannot be negative.');
    end
    if(inv_tol<=0)
        error('#SQUIC: inv_tol must be great than zero.');
    end
    if(term_tol<=0)
        error('#SQUIC: term_tol must be great than zero.');
    end

    % M
    [nrow_M,ncol_M]=size(M);
    if(nrow_M==ncol_M==p)
        error('#SQUIC: M must be square matrix with size pxp.');
    end
    % Make all postive, drop all zeros and force symmetrix
    % zeros droped by defualt
    M=abs(M);
    M = (M + M')/2;

    % X0
    [nrow_X0,ncol_X0]=size(X0);
    if(nrow_X0==ncol_X0==p)
        error('#SQUIC: X0 must be square matrix with size pxp.');
    end

    % W0
    [nrow_W0,ncol_W0]=size(W0);
    if(nrow_W0==ncol_W0==p)
        error('#SQUIC: X0 must be square matrix with size pxp.');
    end

    % Force symmetrix
    X0 = (X0 + X0')/2;
    W0 = (W0 + W0')/2;

    [X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC_MATLAB(Y, lambda, max_iter, inv_tol, term_tol,verbose, M, X0, W0);
end

