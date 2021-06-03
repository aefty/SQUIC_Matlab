function [S,info_times] = SQUIC_S(Y, lambda, verbose, M)
    % SQUIC_S : computes sample covariance matrix.
    % [S,info_times] = SQUIC_S(Y, lambda,verbose, M)
    %
    % input arguments:
    %
    % Y:                Data matrix p x p consisting of p>2 random variables and n>1 samples.
    % l:                Sparsity parameter as a nonzero positive scalar.
    % verbose:          [default 1] Verbosity level as 0 or 1.
    % M:                [default None] Sparsity structure matrix as a sparse p x p matrix.
    %
    % return values:
    %
    % S:                sparse(thresholded) sample covariance matrix.
    % info_times:       List of different compute times.
    %
    %
    % See also SQUIC
    
    [p,~]= size(Y);        

    if(nargin == 2)
        max_iter = 0;
        inv_tol  = 1e-3;
        term_tol = 1e-3;
        verbose  = 1;        
        M        = sparse(p,p);
        %X0       = speye(p,p);
        %W0       = speye(p,p);    
    end
    
    if(nargin == 3)
        max_iter = 0;
        inv_tol  = 1e-3;
        term_tol = 1e-3;
        M        = sparse(p,p);
        %X0       = speye(p,p);
        %W0       = speye(p,p);    
    end
    
    if(nargin == 4)
        max_iter = 0;
        inv_tol  = 1e-3;
        term_tol = 1e-3;
        %X0       = speye(p,p);
        %W0       = speye(p,p);    
    end
        
    [~,S,info_times] = SQUIC(Y, lambda, max_iter, inv_tol, term_tol,verbose,M);
end