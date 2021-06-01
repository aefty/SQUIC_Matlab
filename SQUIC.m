%Arguments:
%1 Y              - samples to build the empirical covariance matrix (pxn)
%2 lambda         - regularization parameter
%3 max_iter       - maximum number of Newton steps
%4 inv_tol       - accuracy of the objective function
%5 term_tol       - accuracy of the objective function
%6 verbose        - initial covariance matrix
%7 M              - bias matrix
%8 X0             - initial precision matrix
%9 W0             - initial inverse precision matrix

function [X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC(Y, lambda, max_iter, inv_tol, term_tol,verbose, M, X0, W0)

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
        inv_tol =  1e-3
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



function [S,info_times] = SQUIC_S(Y, lambda,verbose, M)
    [p,n]= size(Y);
    max_iter=0;
    inv_tol=1e-3;
    term_tol=1e-3;
    X0 = speye(p,p);
    W0 = speye(p,p);
    [~,S,info_times] = SQUIC(Y, lambda, max_iter, inv_tol, term_tol,verbose, M, X0, W0);
end

