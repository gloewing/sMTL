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
## lambda_z>=0: the coefficient support strength sharing coefficient (Zbar penalty)
## maxIter_in: number of inner coordinate descent iterations (within active sets)
## maxIter_out: number of outer coordinate descent iterations
## eigenVec: dummy variable
## WSmethod: dummy variable
## ASpass: dummy variable

using TSVD, Statistics

include("BlockComIHT_inexactAS_opt_old_MT.jl") # IHT algorithm
include("BlockInexactLS_MT.jl") # local search algorithm

# sparse regression with IHT and local search
function BlockComIHT_inexactAS_old_MT(; X::Array{Float64,2},
                    y::Array{Float64,2},
                    rho::Integer,
                    study = nothing, # dummy variable
                    beta::Array{Float64,2},
                    scale::Bool = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0,
                    maxIter::Integer = 5000,
                    localIter = 50,
                    maxIter_in = nothing,
                    maxIter_out = nothing,
                    eig = nothing,
                    eigenVec = nothing, # dummy variable that does nothing
                    WSmethod::Integer = 1, # dummy variable that does nothing
                    ASpass::Bool = false # dummy variable that does nothing
                    )::Array


    n, p = size(X); # number of covaraites
    K = size(y, 2) # num  tasks

    if isnothing(maxIter_in)
        maxIter_in = maxIter
    end

    if isnothing(maxIter_out)
        maxIter_out = maxIter
    end

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        Xsd = std(X, dims=1) .* (n - 1) / n; # glmnet style MLE of sd
        sdMat = Xsd; # save std of covariates of ith study in ith row of matrix
        X .= X ./ Xsd; # standardize ith study's covariates
        #

        sdMat = hcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat'; # current solution β

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)
        eig = 0;
        eig = tsvd(X)[2][1]; # max eigenvalue of X^T X

    else
        eig = Float64(eig)
    end

    L = eig^2 * sqrt(K) / n # L without regularization terms (updated in optimization fn below)

    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    for v = 1:vals
        # use warm starts as previous value
        beta = BlockComIHT_inexactAS_opt_old_MT(X = X,
                                        y = y,
                                        rho = rho,
                                        B = beta,
                                        K = K,
                                        L = L,
                                        n = n,
                                        maxIter_in = maxIter_in,
                                        maxIter_out = maxIter_out,
                                        lambda1 = Float64(lambda1[v]),
                                        lambda2 = Float64(lambda2[v]),
                                        lambda_z = Float64(lambda_z[v]),
                                        p = p
                                        )
        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockInexactLS_MT(X = X,
                                y = y,
                                s = rho,
                                beta = beta,
                                lambda1 = Float64(lambda1[v]),
                                lambda2 = Float64(lambda2[v]),
                                lambda_z = Float64(lambda_z[v]),
                                K = K,
                                n = n,
                                p = p,
                                maxIter = localIter[v] )
        end


        if scale
            βmat[:,:,v] = beta ./ sdMat'; # rescale by sd
        else
            βmat[:,:,v] = beta
        end
    end

    if vals == 1
        # if only one tuning value, just return a matrix
        return βmat[:,:,1];
    else
        return βmat;
    end

end
