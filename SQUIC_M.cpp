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
#define integer long
// SQUIC Library iterface

extern "C"
{

    void SQUIC_CPP(
        int mode,
        integer p,
        integer n_train, double *Y_train,
        integer n_test, double *Y_test,
        double lambda,
        integer *M_rinx, integer *M_cptr, double *M_val, integer M_nnz,
        int max_iter, double drop_tol, double term_tol, int verbose,
        integer *&X_rinx, integer *&X_cptr, double *&X_val, integer &X_nnz,
        integer *&W_rinx, integer *&W_cptr, double *&W_val, integer &W_nnz,
        int &info_num_iter,
        double *&info_times,     //length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
        double *&info_objective, // length must be size max_iter
        double &info_dgap,
        double &info_logdetx,
        double &info_trXS_test);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs < 5)
    {
        mexErrMsgIdAndTxt("SQUIC_M:arguments",
                          "Missing arguments, please specify\n"
                          "             data_train     - data maximum likelihood (pxn)\n"
                          "             lambda         - regularization parameter\n"
                          "             max_iter       - maximum number of Newton steps\n"
                          "             drop_tol       - accuracy of the objective function\n"
                          "             term_tol       - accuracy of the objective function\n"
                          "             verbose        - (optional) initial covariance matrix\n"
                          "             mode           - (optional) run mode of SQUIC (block =0, scalar=5)\n"
                          "             M              - (optional) bias matrix\n"
                          "             X0             - (optional) initial precision matrix\n"
                          "             W0             - (optional) initial inverse precision matrix\n"
                          "             data_test      - (optional) data to compute likelihood\n");
    }

    int argIdx = 0;

    ////////////////////////////////////////
    // 0. data_train matrix: double dense matrix
    ////////////////////////////////////////
    if (!mxIsDouble(prhs[argIdx]))
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a double matrix. (Arg. %d)",
                          argIdx + 1);
    }

    double *data_train = mxGetPr(prhs[argIdx]);
    integer p = mxGetM(prhs[argIdx]);
    integer n_train = mxGetN(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 1. lambda: scalar double
    ////////////////////////////////////////
    if (!mxIsNumeric(prhs[argIdx]) || mxGetM(prhs[argIdx]) != mxGetN(prhs[argIdx]) || mxGetN(prhs[argIdx]) != 1) //  Scalar (1x1) Numeric
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar. (Arg. %d)",
                          argIdx + 1);
    }

    double lambda = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 2. max_iter: scalar interger
    ////////////////////////////////////////
    if (!mxIsNumeric(prhs[argIdx]) || mxGetM(prhs[argIdx]) != mxGetN(prhs[argIdx]) || mxGetN(prhs[argIdx]) != 1) //  Scalar (1x1) Numeric
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar non-negative integer. (Arg. %d)",
                          argIdx + 1);
    }

    if (std::floor(mxGetScalar(prhs[argIdx])) != mxGetScalar(prhs[argIdx]))
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar non-negative integer. (Arg. %d)",
                          argIdx + 1);
    }

    if (mxGetScalar(prhs[argIdx]) < 0)
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar non-negative integer. (Arg. %d)",
                          argIdx + 1);
    }

    int max_iter = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 3. drop_tol: scalar double
    ////////////////////////////////////////
    if (!mxIsNumeric(prhs[argIdx]) || mxGetM(prhs[argIdx]) != mxGetN(prhs[argIdx]) || mxGetN(prhs[argIdx]) != 1) // Scalar (1x1) Numeric
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar. (Arg. %d)",
                          argIdx + 1);
    }
    double drop_tol = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 4. term_tol: scalar double
    ////////////////////////////////////////
    if (!mxIsNumeric(prhs[argIdx]) || mxGetM(prhs[argIdx]) != mxGetN(prhs[argIdx]) || mxGetN(prhs[argIdx]) != 1) // Scalar (1x1) Numeric
    {
        mexErrMsgIdAndTxt("SQUIC:type",
                          "Expected a scalar. (Arg. %d)",
                          argIdx + 1);
    }
    double term_tol = mxGetScalar(prhs[argIdx]);
    argIdx++;

    ////////////////////////////////////////
    // 5. verbose - optional
    ////////////////////////////////////////
    int verbose = 1;
    if (nrhs > argIdx)
    {
        if (!mxIsNumeric(prhs[argIdx]) || mxGetM(prhs[argIdx]) != mxGetN(prhs[argIdx]) || mxGetN(prhs[argIdx]) != 1) //  Scalar (1x1) Numeric
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "Expected a scalar. (Arg. %d)",
                              argIdx + 1);
        }

        verbose = mxGetScalar(prhs[argIdx]);
        argIdx++;
    }

    ////////////////////////////////////////
    // 6. mode - optional
    ////////////////////////////////////////
    int mode = 0;
    if (nrhs > argIdx)
    {

        if (!mxIsNumeric(prhs[argIdx]) || mxGetScalar(prhs[argIdx]) > 9 || 0 > mxGetScalar(prhs[argIdx])) //  Scalar (1x1) Numeric
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "Expected a scalar integer from 0 to 9. (Arg. %d)",
                              argIdx + 1);
        }

        mode = mxGetScalar(prhs[argIdx]);
        argIdx++;
    }

    /////////////////////////////
    // 7. M matrix: sparse matrix - Optional
    /////////////////////////////
    integer *M_i;
    integer *M_j;
    double *M_val;
    integer M_nnz = 0;
    if (nrhs > argIdx)
    {
        mxArray *M = (mxArray *)prhs[argIdx];
        integer M_mrows = mxGetM(M);
        integer M_ncols = mxGetN(M);

        M_nnz = mxGetNzmax(M);

        if (M_mrows != M_ncols)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "M must be a square matrix.");
        }
        if (!mxIsSparse(M))
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "M must be in sparse format.");
        }
        if (M_mrows != p)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "M must have size corresponding to Y");
        }

        //!! Note matlab empty sparse matrix have seem to alway have 1 nnz with a value of 0.0)
        if (M_nnz > 0) // M Matrix NOT zeros
        {
            // Copy MATLAB matrix to raw buffer;
            mwIndex *M_i_matlab = mxGetIr(M);
            mwIndex *M_j_matlab = mxGetJc(M);
            double *M_val_matlab = mxGetPr(M);

            // Special case where we have empty sparse matrix (nnz is still =1)
            if (M_nnz == 1 && M_val_matlab[0] == 0.0)
            {
                M_nnz = 0;
            }
            else
            {
                M_i = new integer[M_nnz];
                M_j = new integer[p + 1];
                M_val = new double[M_nnz];

                // copy i and val
                for (integer i = 0; i < M_nnz; i++)
                {
                    M_i[i] = M_i_matlab[i];
                    M_val[i] = M_val_matlab[i];
                }
                // copy column count
                for (integer i = 0; i < p + 1; i++)
                {
                    M_j[i] = M_j_matlab[i];
                }
            }
        }
        argIdx++;
    }

    /////////////////////////////
    // 7. X0 matrix: sparse matrix - Optional
    /////////////////////////////
    integer *X_i;
    integer *X_j;
    double *X_val;
    integer X_nnz = 0;

    if (nrhs > argIdx)
    {
        mxArray *X = (mxArray *)prhs[argIdx];
        integer X_mrows = mxGetM(X);
        integer X_ncols = mxGetN(X);
        X_nnz = mxGetNzmax(X);

        if (X_mrows != X_ncols)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "X0 must be a square matrix.");
        }
        if (!mxIsSparse(X))
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "X0 must be in sparse format.");
        }
        if (X_mrows != p)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "X0 must have size corresponding to Y");
        }

        if (X_nnz > 0) // M Matrix NOT zeros
        {
            // Copy MATLAB matrix to raw buffer;
            mwIndex *X_i_matlab = mxGetIr(X);
            mwIndex *X_j_matlab = mxGetJc(X);
            double *X_val_matlab = mxGetPr(X);

            // Special case where we have empty sparse matrix (nnz is still =1)
            if (X_nnz == 1 && X_val_matlab[0] == 0.0)
            {
                X_nnz = 0;
            }
            else
            {
                X_i = new integer[X_nnz];
                X_j = new integer[p + 1];
                X_val = new double[X_nnz];

                for (integer i = 0; i < X_nnz; i++)
                {
                    X_i[i] = X_i_matlab[i];
                    X_val[i] = X_val_matlab[i];
                }

                for (integer i = 0; i < p + 1; i++)
                {
                    X_j[i] = X_j_matlab[i];
                }
            }
        }
        argIdx++;
    }

    /////////////////////////////
    // 8. W matrix: sparse matrix - Optional
    /////////////////////////////
    integer *W_i;
    integer *W_j;
    double *W_val;
    integer W_nnz = 0;

    if (nrhs > argIdx)
    {
        mxArray *W = (mxArray *)prhs[argIdx];
        integer W_mrows = mxGetM(W);
        integer W_ncols = mxGetN(W);
        W_nnz = mxGetNzmax(W);

        if (W_mrows != W_ncols)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "W0 must be a square matrix.");
        }
        if (!mxIsSparse(W))
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "W0 must be in sparse format.");
        }
        if (W_mrows != p)
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "W0 must have size corresponding to Y");
        }

        if (W_nnz > 0) // W Matrix NOT zeros
        {
            // Copy MATLAB matrix to raw buffer;
            mwIndex *W_i_matlab = mxGetIr(W);
            mwIndex *W_j_matlab = mxGetJc(W);
            double *W_val_matlab = mxGetPr(W);

            // Special case where we have empty sparse matrix (nnz is still =1)
            if (W_nnz == 1 && W_val_matlab[0] == 0.0)
            {
                W_nnz = 0;
            }
            else
            {
                W_i = new integer[W_nnz];
                W_j = new integer[p + 1];
                W_val = new double[W_nnz];

                for (integer i = 0; i < W_nnz; i++)
                {
                    W_i[i] = W_i_matlab[i];
                    W_val[i] = W_val_matlab[i];
                }

                for (integer i = 0; i < p + 1; i++)
                {
                    W_j[i] = W_j_matlab[i];
                }
            }
        }
        argIdx++;
    }

    ////////////////////////////////////////
    // 9. test_data: dense double matrix
    ////////////////////////////////////////
    double *data_test;
    integer n_test = 0;
    if (nrhs > argIdx)
    {

        if (!mxIsDouble(prhs[argIdx]))
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "Expected a numeric matrix. (Arg. %d)",
                              argIdx + 1);
        }

        data_test = mxGetPr(prhs[argIdx]);
        n_test = mxGetN(prhs[argIdx]);

        mexPrintf("mxGetM(prhs[argIdx])", mxGetM(prhs[argIdx]));

        if (mxGetM(prhs[argIdx]) != p) // data_test is not correct
        {
            mexErrMsgIdAndTxt("SQUIC:type",
                              "test_data must have p rows");
        }
        argIdx++;
    }

    /////////////////////////
    // Run SQUIC
    /////////////////////////
    int info_num_iter = -1;                                     // Number of Newten steps required by SQUIC
    double info_dgap = -1e-12;                                  // Duality Gap between primal and dual
    double info_logdetx = -1e-12;                               // Can be used for likelilook (AIC or BIC) computation of test data
    double info_trXS_test = -1e-12;                             // Can be used for likelilook (AIC or BIC) computation of test data
    double *info_times = new double[6];                         // This need to be of size 6
    double *info_objective = new double[std::max(1, max_iter)]; // The objective value list, must be of size max(max_iter,1). If max_iter=0, we still keep this with size of 1

    // Run SQUIC
    SQUIC_CPP(
        mode,
        p,
        n_train, data_train,
        n_test, data_test,
        lambda,
        M_i, M_j, M_val, M_nnz,
        max_iter, drop_tol, term_tol, verbose,
        X_i, X_j, X_val, X_nnz,
        W_i, W_j, W_val, W_nnz,
        info_num_iter,
        info_times,
        info_objective,
        info_dgap,
        info_logdetx,
        info_trXS_test);

    double time_total = info_times[0];
    double time_impcov = info_times[1];
    double time_optimz = info_times[2];
    double time_factor = info_times[3];
    double time_aprinv = info_times[4];
    double time_updte = info_times[5];
    double *objective = info_objective;
    double duality_gap = info_dgap;
    double logdetX = info_logdetx;
    double trXS_test = info_trXS_test;

    int ifield;
    int field_size;
    mxArray *field, *tmp;
    mwIndex *ia, *ja;
    double *pr, *a;

    // Transfer ouput
    argIdx = 0;
    if (nlhs > argIdx)
    {
        ////////////////////////////////////////
        // OUT 0. X
        ////////////////////////////////////////

        mxArray *X_out = mxCreateSparse((mwSize)p, (mwSize)p, (mwSize)X_nnz, mxREAL);
        mwIndex *X_out_i = mxGetIr(X_out);
        mwIndex *X_out_j = mxGetJc(X_out);
        double *X_out_val = mxGetPr(X_out);

        // Clear Vector
        for (integer i = 0; i < p + 1; i++)
        {
            X_out_j[i] = X_j[i];
        }

        // Write matrix
        for (integer i = 0; i < X_nnz; i++)
        {
            X_out_i[i] = X_i[i];
            X_out_val[i] = X_val[i];
        }

        plhs[argIdx] = X_out;
        argIdx++;
    }

    if (nlhs > argIdx)
    {
        ////////////////////////////////////////
        // OUT 1. W
        ////////////////////////////////////////
        argIdx = 1;
        mxArray *W_out = mxCreateSparse((mwSize)p, (mwSize)p, (mwSize)W_nnz, mxREAL);
        mwIndex *W_out_i = mxGetIr(W_out);
        mwIndex *W_out_j = mxGetJc(W_out);
        double *W_out_val = mxGetPr(W_out);

        // Copy matrix
        for (integer i = 0; i < p + 1; i++)
        {
            W_out_j[i] = W_j[i];
        }
        // Write matrix
        for (integer i = 0; i < W_nnz; i++)
        {
            W_out_i[i] = W_i[i];
            W_out_val[i] = W_val[i];
        }

        plhs[argIdx] = W_out;
        argIdx++;
    }

    ////////////////////////////////////////
    // OUT 2. info
    ////////////////////////////////////////

    if (nlhs > argIdx)
    {

        if (max_iter == 0) // Special max_iter==0: SQUIC only compute the sparse sample covariance S
        {
            ////////////////////////////////////////
            // OUT 1. info
            ////////////////////////////////////////
            const char *statnames[] = {
                "time_total",
                "time_impcov"};

            plhs[argIdx] = mxCreateStructMatrix((mwSize)1, (mwSize)1, 2, statnames);

            // OUT 1.1 time_total
            ifield = 0;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_total;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            // OUT 1.2 time_impcov
            ifield++;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_impcov;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);
            argIdx++;
        }
        else // Regular case
        {

            if (n_test > 0)
            {

                const char *statnames[] = {
                    "time_total",
                    "time_impcov",
                    "time_optimz",
                    "time_factor",
                    "time_aprinv",
                    "time_updte",
                    "objective",
                    "duality_gap",
                    "logdetX",
                    "trXS_test"};
                plhs[argIdx] = mxCreateStructMatrix((mwSize)1, (mwSize)1, 10, statnames);
            }
            else
            {
                const char *statnames[] = {
                    "time_total",
                    "time_impcov",
                    "time_optimz",
                    "time_factor",
                    "time_aprinv",
                    "time_updte",
                    "objective",
                    "duality_gap",
                    "logdetX"};
                plhs[argIdx] = mxCreateStructMatrix((mwSize)1, (mwSize)1, 9, statnames);
            }

            // OUT 2.0 time_total
            ifield = 0;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_total;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            // OUT 2.1 time_impcov
            ifield = 1;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_impcov;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            // OUT 2.2 time_optimz
            ifield = 2;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_optimz;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT  2.3 time_factor
            ifield = 3;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_factor;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT 2.4 time_aprinv
            ifield = 4;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_aprinv;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT 2.5 time_updte
            ifield = 5;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = time_updte;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT 2.6 objective
            ifield = 6;
            field_size = info_num_iter;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            for (integer i = 0; i < field_size; i++)
            {
                pr[i] = objective[i];
            }
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT 2.7 duality_gap
            ifield = 7;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = duality_gap;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            //OUT 2.8 logdetX
            ifield = 8;
            field_size = 1;
            field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
            pr = mxGetPr(field);
            pr[0] = logdetX;
            mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);

            if (n_test > 0)
            {
                //OUT 2.9 trXS_test
                ifield = 9;
                field_size = 1;
                field = mxCreateDoubleMatrix((mwSize)1, (mwSize)field_size, mxREAL);
                pr = mxGetPr(field);
                pr[0] = trXS_test;
                mxSetFieldByNumber(plhs[argIdx], (mwSize)0, ifield, field);
            }

            argIdx++;
        }
    }

    //clean up
    if (M_nnz > 0)
    {
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

    delete[] info_times;
    delete[] info_objective;

    return;
}
