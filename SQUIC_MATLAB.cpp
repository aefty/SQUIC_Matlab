// This is the MEX wrapper for SQUIC

#include <mex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <cmath>

// SQUIC is fixed with type: long in (This is a requirement for Cholmod)


extern "C"
{
    void SQUIC_CPP(
        int mode,
        long p,
        long n, double *Y,
        double lambda,
        long *M_rinx, long *M_cptr, double *M_val, long M_nnz,
        int max_iter, double inv_tol, double term_tol, int verbose,
        long *&X_rinx, long *&X_cptr, double *&X_val, long &X_nnz,
        long *&W_rinx, long *&W_cptr, double *&W_val, long &W_nnz,
        int &info_num_iter,
        double *&info_times,      //length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
        double *&info_objective, // length must be size max_iter
        double &info_logdetX,
        double &info_trSX);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (nrhs < 5) {
        mexErrMsgIdAndTxt("SQUIC_M:arguments",
                          "Missing arguments, please specify\n"
                          "             Y              - data maximum likelihood (pxn)\n"
                          "             lambda         - regularization parameter\n"
                          "             max_iter       - maximum number of Newton steps\n"
                          "             inv_tol        - accuracy of the objective function\n"
                          "             term_tol       - accuracy of the objective function\n"
                          "             verbose        - (optional) initial covariance matrix\n"
                          "             M              - penalty matrix\n"
                          "             X0             - initial precision matrix\n"
                          "             W0             - initial inverse precision matrix\n");
    }

    int argIdx = 0;

    ////////////////////////////////////////
    // 0. Y matrix: double dense matrix
    ////////////////////////////////////////
    double *Y = mxGetPr(prhs[argIdx]);
    long p = mxGetM(prhs[argIdx]);
    long n = mxGetN(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 1. lambda: scalar double
    ////////////////////////////////////////
    double lambda = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 2. max_iter: scalar interger
    ////////////////////////////////////////
    int max_iter = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 3. drop_tol: scalar double
    ////////////////////////////////////////
    double inv_tol = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 4. term_tol: scalar double
    ////////////////////////////////////////
    double term_tol = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 5. verbose
    ////////////////////////////////////////
    int verbose = mxGetScalar(prhs[argIdx]);
    argIdx++;

    /////////////////////////////
    // 6. M matrix: sparse matrix
    /////////////////////////////
    long *M_i;
    long *M_j;
    double *M_val;
    long M_nnz = 0;

    mxArray *M = (mxArray *)prhs[argIdx];
    long M_mrows = mxGetM(M);
    long M_ncols = mxGetN(M);

    M_nnz = mxGetNzmax(M);

    //!! Note matlab empty sparse matrix seems to alway have 1 nnz with a value of 0.0)
    if (M_nnz > 0) { // M Matrix NOT zeros

        // Copy MATLAB matrix to raw buffer;
        mwIndex *M_i_matlab = mxGetIr(M);
        mwIndex *M_j_matlab = mxGetJc(M);
        double *M_val_matlab = mxGetPr(M);

        // Special case where we have empty sparse matrix (nnz is still =1 but
        // the M_j_matlab will be all zeros only check the last value )
        if (M_nnz == 1 && M_j_matlab[p] == 0) {
            M_nnz = 0;
        } else {
            M_i = new long[M_nnz];
            M_j = new long[p + 1];
            M_val = new double[M_nnz];

            // copy i and val
            for (long i = 0; i < M_nnz; i++) {
                M_i[i] = M_i_matlab[i];
                M_val[i] = M_val_matlab[i];
            }
            // copy column count
            for (long i = 0; i < p + 1; i++) {
                M_j[i] = M_j_matlab[i];
            }
        }
    }
    argIdx++;


    /////////////////////////////
    // 7. X0 matrix: sparse matrix
    /////////////////////////////
    long *X_i;
    long *X_j;
    double *X_val;
    long X_nnz = 0;

    mxArray *X = (mxArray *)prhs[argIdx];
    long X_mrows = mxGetM(X);
    long X_ncols = mxGetN(X);
    X_nnz = mxGetNzmax(X);

    if (X_nnz > 0) { // M Matrix NOT zeros
        // Copy MATLAB matrix to raw buffer;
        mwIndex *X_i_matlab = mxGetIr(X);
        mwIndex *X_j_matlab = mxGetJc(X);
        double *X_val_matlab = mxGetPr(X);

        // Special case where we have empty sparse matrix (nnz is still =1)
        if (X_nnz == 1 && X_j_matlab[0] == 0) {
            X_nnz = 0;
        } else {
            X_i = new long[X_nnz];
            X_j = new long[p + 1];
            X_val = new double[X_nnz];

            for (long i = 0; i < X_nnz; i++) {
                X_i[i] = X_i_matlab[i];
                X_val[i] = X_val_matlab[i];
            }

            for (long i = 0; i < p + 1; i++) {
                X_j[i] = X_j_matlab[i];
            }
        }
    }
    argIdx++;


    /////////////////////////////
    // 8. W0 matrix: sparse matrix
    /////////////////////////////
    long *W_i;
    long *W_j;
    double *W_val;
    long W_nnz = 0;

    mxArray *W = (mxArray *)prhs[argIdx];
    long W_mrows = mxGetM(W);
    long W_ncols = mxGetN(W);
    W_nnz = mxGetNzmax(W);

    if (W_nnz > 0) { // W Matrix NOT zeros

        // Copy MATLAB matrix to raw buffer;
        mwIndex *W_i_matlab = mxGetIr(W);
        mwIndex *W_j_matlab = mxGetJc(W);
        double *W_val_matlab = mxGetPr(W);

        // Special case where we have empty sparse matrix (nnz is still =1)
        if (W_nnz == 1 && W_j_matlab[0] == 0) {
            W_nnz = 0;
        } else {
            W_i = new long[W_nnz];
            W_j = new long[p + 1];
            W_val = new double[W_nnz];

            for (long i = 0; i < W_nnz; i++) {
                W_i[i] = W_i_matlab[i];
                W_val[i] = W_val_matlab[i];
            }

            for (long i = 0; i < p + 1; i++) {
                W_j[i] = W_j_matlab[i];
            }
        }
    }
    argIdx++;

    // Default Result Values
    int    info_num_iter = 0;
    double info_logdetX = 0.0;
    double info_trSX = 0.0;
    double *info_times_buffer = new double[6];
    double *info_objective_buffer = new double[std::max(1, max_iter)];

    // hardcode mode
    int mode = 0;

    // Run SQUIC
    SQUIC_CPP(
        mode,
        p,
        n, Y,
        lambda,
        M_i, M_j, M_val, M_nnz,
        max_iter, inv_tol, term_tol, verbose,
        X_i, X_j, X_val, X_nnz,
        W_i, W_j, W_val, W_nnz,
        info_num_iter,
        info_times_buffer,
        info_objective_buffer,
        info_logdetX,
        info_trSX);

    // Transfer ouput
    argIdx = 0;

    ////////////////////////////////////////
    // OUT X
    ////////////////////////////////////////
    if (nlhs > argIdx) {
        mxArray *X_out = mxCreateSparse((mwSize)p, (mwSize)p, (mwSize)X_nnz, mxREAL);
        mwIndex *X_out_i = mxGetIr(X_out);
        mwIndex *X_out_j = mxGetJc(X_out);
        double *X_out_val = mxGetPr(X_out);

        // Clear Vector
        for (long i = 0; i < p + 1; i++) {
            X_out_j[i] = X_j[i];
        }

        // Write matrix
        for (long i = 0; i < X_nnz; i++) {
            X_out_i[i] = X_i[i];
            X_out_val[i] = X_val[i];
        }

        plhs[argIdx] = X_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT W
    ////////////////////////////////////////
    if (nlhs > argIdx) {
        mxArray *W_out = mxCreateSparse((mwSize)p, (mwSize)p, (mwSize)W_nnz, mxREAL);
        mwIndex *W_out_i = mxGetIr(W_out);
        mwIndex *W_out_j = mxGetJc(W_out);
        double *W_out_val = mxGetPr(W_out);

        // Copy matrix
        for (long i = 0; i < p + 1; i++) {
            W_out_j[i] = W_j[i];
        }
        // Write matrix
        for (long i = 0; i < W_nnz; i++) {
            W_out_i[i] = W_i[i];
            W_out_val[i] = W_val[i];
        }

        plhs[argIdx] = W_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT info_times_buffer
    ////////////////////////////////////////
    if (nlhs > argIdx) {
        mxArray *info_times_out = mxCreateDoubleMatrix((mwSize)1, (mwSize)6, mxREAL);
        double *ptr = mxGetPr(info_times_out);

        for (int i = 0; i < 6; i++) {
            ptr[i] = info_times_buffer[i];
        }

        plhs[argIdx] = info_times_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT info_objective_buffer
    ////////////////////////////////////////
    if (nlhs > argIdx) {

        mxArray *info_objective_out = mxCreateDoubleMatrix((mwSize)1, (mwSize)info_num_iter, mxREAL);
        double *ptr = mxGetPr(info_objective_out);

        for (int i = 0; i < info_num_iter; i++) {
            ptr[i] = info_objective_buffer[i];
        }

        plhs[argIdx] = info_objective_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT info_logdetX
    ////////////////////////////////////////
    if (nlhs > argIdx) {
        mxArray *info_logdetX_out = mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL);
        double *ptr = mxGetPr(info_logdetX_out);

        ptr[0] = info_logdetX;
        plhs[argIdx] = info_logdetX_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT info_trSX
    ////////////////////////////////////////
    if (nlhs > argIdx) {
        mxArray *info_trSX_out = mxCreateDoubleMatrix((mwSize)1, (mwSize)info_num_iter, mxREAL);
        double *ptr = mxGetPr(info_trSX_out);

        ptr[0] = info_trSX;
        plhs[argIdx] = info_trSX_out;
        argIdx++;
    }

    //clean up
    if (M_nnz > 0) {
        delete[] M_i;
        delete[] M_j;
        delete[] M_val;
    }

    delete[] X_i;
    delete[] X_j;
    delete[] X_val;

    delete[] W_i;
    delete[] W_j;
    delete[] W_val;

    delete[] info_times_buffer;
    delete[] info_objective_buffer;

    return;
}
