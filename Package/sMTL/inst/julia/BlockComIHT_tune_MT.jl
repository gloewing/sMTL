# Model Fitting Code
## X: n x p design matrix (feature matrix)
## y: n x K outcome matrix
## rho: Sparsity level (integer)
## beta: p*K initial solution
## scale: whether to center scale features
## eig: max eigenvalue of all design matrices
## idx: active set indices
## lambda1>=0: the ridge coefficient
## lambda2>=0: the coefficient value strength sharing coefficient (Bbar penalty)
## maxIter: number of max coordinate descent iterations
## localIter: max number of local search iterations

using TSVD, Statistics

include("BlockComIHT_opt_MT.jl") # IHT
include("BlockComIHT_opt_cvx_MT.jl") # IHT convex
include("BlockLS_MT.jl") # local search

# sparse regression with IHT
function BlockComIHT_MT(; X::Array{Float64,2},
                    y::Array{Float64,2},
                    rho::Integer,
                    study = nothing, # dummy variable
                    beta::Array{Float64,2} = 0,
                    scale::Bool = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    maxIter::Integer = 5000,
                    localIter = 50,
                    eig = nothing
                    )::Array

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # max eigenvalue for Lipschitz constant
    # localIter is a vector as long as lambda1/lambda2 and specifies the number of local search iterations for each lambda

    n, p = size(X); # number of covaraites
    localIter = Int.(localIter);
    K = size(y, 2) #length( unique(study) ); # number of tasks
    indxList = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        Xsd = std(X, dims=1) .* (n - 1) / n; # glmnet style MLE of sd
        sdMat = Xsd; # save std of covariates of ith study in ith row of matrix
        X .= X ./ Xsd; # standardize ith study's covariates

        sdMat = hcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat'; # current solution β

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)
        eig = tsvd(X)[2][1]; # max eigenvalue of X^T X

    else
        eig = Float64(eig)
    end

    L = eig^2 * K / n # L without regularization terms (updated in optimization fn below)

    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    if rho < p
        ################
        # sparse setting
        ################
        for v = 1:vals
            ###############
            # IHT
            ###############
            # use warm starts as previous value
            beta = BlockComIHT_opt_MT(X = X,
                                            y = y,
                                            rho = rho,
                                            βhat = beta,
                                            K = K,
                                            L = L,
                                            n = n,
                                            maxIter = maxIter,
                                            lambda1 = Float64(lambda1[v]),
                                            lambda2 = Float64(lambda2[v]),
                                            p = p
                                            )
            ###############
            # local search
            ###############
            if localIter[v] > 0
                # run local search if positive number of local search iterations for this lambda1/lambda2 value
                beta = BlockLS_MT(X = X,
                                y = y,
                                s = rho,
                                B = beta,
                                K = K,
                                n = n,
                                lambda1 = Float64(lambda1[v]),
                                lambda2 = Float64(lambda2[v]),
                                p = p,
                                maxIter = localIter[v])
            end

            #############################################
            # rescale betas back to original scale
            #############################################
            if scale
                βmat[:,:,v] = beta ./ sdMat'; # rescale by sd
            else
                βmat[:,:,v] = beta
            end
        end

    else
        ################
        # convex setting
        ################
        for v = 1:vals
            ###############
            # IHT
            ###############
            # use warm starts as previous value
            beta = BlockComIHT_opt_cvx_MT(X = X,
                                            y = y,
                                            βhat = beta,
                                            K = K,
                                            L = L,
                                            n = n,
                                            maxIter = maxIter,
                                            lambda1 = Float64(lambda1[v]),
                                            lambda2 = Float64(lambda2[v]),
                                            p = p
                                            )

            #############################################
            # rescale betas back to original scale
            #############################################
            if scale
                βmat[:,:,v] = beta ./ sdMat'; # rescale by sd
            else
                βmat[:,:,v] = beta
            end
        end

    end

    if vals == 1
        # if only one tuning value, just return a matrix
        return βmat[:,:,1];
    else
        return βmat;
    end

end
