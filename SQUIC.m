function [iC,C,stats] = SQUIC(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option, M_option,X0_option,W0_option,data_test_option)

% Arguments:
%data_train     - samples to build the empirical covariance matrix (pxn)
%lambda         - regularization parameter
%max_iter       - maximum number of Newton steps
%drop_tol       - accuracy of the objective function
%term_tol       - accuracy of the objective function
%verbose        - initial covariance matrix
%mode           - run mode of SQUIC (block =0, scalar=5)
%M              - bias matrix
%X0             - initial precision matrix
%W0             - initial inverse precision matrix
% data_test     - initial precision matrix



if( nargin == 6) 
    mode_option = 0;
    [iC,C,stats]=SQUIC_M(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option);
end

if( nargin == 7) 
    [iC,C,stats]=SQUIC_M(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option);
end

if( nargin == 8 ) 
    if(issymmetric(M_option))
        [iC,C,stats]=SQUIC_M(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option,M_option);
    else
        error('M must be symmetric');
    end
end

if(nargin == 9 ) 
    % Error
    error('Both X0 and W0 must be provided');
end

if( nargin == 10 )
    if(issymmetric(M_option) || issymmetric(X0_option) || issymmetric(W0_option) )
        [iC,C,stats]=SQUIC_M(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option,M_option,X0_option,W0_option);
    else
        error('M, X0 and W0 must be symmetric');
    end
end


if( nargin == 11 )
    if(issymmetric(M_option) && issymmetric(X0_option) && issymmetric(W0_option) )
        [iC,C,stats]=SQUIC_M(data_train,lambda,max_iter,drop_tol,term_tol,verbose,mode_option,M_option,X0_option,W0_option,data_test_option);
    else
        error('M, X0 and W0 must be symmetric');
    end
end
    
end

                                         
