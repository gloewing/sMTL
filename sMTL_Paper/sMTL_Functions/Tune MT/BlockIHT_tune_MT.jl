using TSVD, Statistics

include("BlockIHT_opt_MT.jl")
include("BlockLS_MT.jl") # local search algorithm -- same as with lambda_z but set to 0 always

# sparse regression with IHT -- Block IHT
function BlockIHT(; X,
                    y,
                    rho,
                    study = nothing, # dummy variable
                    beta = 0,
                    scale = true,
                    lambda = 0,
                    maxIter = 5000,
                    localIter = 50,
                    eig = nothing
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # max eigenvalue for Lipschitz constant
    # localIter is number of local search iterations

    y = Matrix(y);
    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Matrix(beta); # initial value
    rho = Int64(rho);
    K = size(y, 2); # number of tasks

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
        eig = 0;
        eig = tsvd(X)[2][1]; # max eigenvalue of X^T X
    else
        eig = Float64(eig)
    end

    L = sqrt(K) * eig^2 / n

    ##############################
    # optimization
    vals = length(lambda)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end


    for v = 1:vals
        # use warm starts as previous value
        beta = BlockIHT_opt_MT(X = X,
                                        y = y,
                                        rho = rho,
                                        beta = beta,
                                        K = K,
                                        L = L,
                                        n = n,
                                        maxIter = maxIter,
                                        lambda = lambda[v],
                                        p = p
        )

        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockLS_MT(X = X,
                            y = y,
                            rho = rho,
                            # indxList = indxList,
                            B = beta,
                            K = K,
                            n = n,
                            lambda1 = lambda[v],
                            lambda2 = 0,
                            p = p,
                            maxIter = localIter[v])
        end

        if scale
            βmat[:,:,v] = beta ./ sdMat'; # rescale by sd
        else
            βmat[:,:,v] = beta
        end
    end

    ##############################
    # return matrix (if one tuning value) or array (if multiple lambdas)

    if vals == 1
        # if only one tuning value, just return a matrix
        return βmat[:,:,1];
    else
        return βmat;
    end

end
