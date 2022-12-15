# Model Fitting Code
## X: n x p design matrix (feature matrix)
## y: n x 1 outcome vector
## rho: Sparsity level (integer)
## study: dummy variable
## beta: p*K initial solution
## scale: whether to center scale features
## lambda>=0: the ridge coefficient
## maxIter: number of max coordinate descent iterations
## localIter: max number of local search iterations
## eig: max eigenvalue for Lipschitz constant

using TSVD, Statistics
include("BlockIHT_opt.jl")
include("BlockLS.jl") # local search algorithm -- same as with lambda_z but set to 0 always

# sparse regression with IHT -- Block IHT
function BlockIHT(; X::Array{Float64,2},
                    y::Array{Float64,1},
                    rho::Integer,
                    study::Array{Int64,1},
                    beta::Array{Float64,2} = 0,
                    scale::Bool = true,
                    lambda = 0,
                    maxIter::Integer = 1000,
                    localIter = 50,
                    eig = nothing
                    )::Array

    n, p = size(X); # number of covaraites
    K = length( unique(study) ); # number of studies
    indxList = [Vector{Int64}() for i in 1:K]; # list of vectors of indices of studies
    nVec = Vector{Int}(undef, K)

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p, K); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            indxList[i] = indx; # save indices
            n_k = length(indx); # study k sample size
            nVec[i] = n_k; # save sample size
            Xsd = std(X[indx,:], dims=1) .* (n_k - 1) / n_k; # glmnet style MLE of sd
            sdMat[:,i] = Xsd[1,:]; # save std of covariates of ith study in ith row of matrix
            X[indx,:] .= X[indx,:] ./ Xsd; # standardize ith study's covariates
        end

        sdMat = vcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat; # current solution β

    else
        # otherwise just make this a vector of ones for multiplication
        # by coefficient estimates later

        for i = 1:K
            indxList[i] = findall(x -> x == i, study); # indices of rows for ith study
            indx = indxList[i];
            n_k = length(indx); # study k sample size
            nVec[i] = n_k; # save sample size
        end

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)
        eig = 0;
        # if not provided by user
        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            a2 = tsvd(X[indx,:])[2][1]; # max eigenvalue of X^T X
            if (a2 > eig)
                eig = a2
            end
        end
    else
        eig = Float64(eig)
    end

    L = sqrt(K) * eig^2 / maximum(nVec) # L without the regularization term which is added in opt function

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
        beta = BlockIHT_opt(X = X,
                                        y = y,
                                        rho = rho,
                                        indxList = indxList,
                                        beta = beta,
                                        K = K,
                                        L = L,
                                        nVec = nVec,
                                        maxIter = maxIter,
                                        lambda = Float64(lambda[v]),
                                        p = p
        )

        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockLS(X = X,
                            y = y,
                            s = rho,
                            indxList = indxList,
                            B = beta,
                            K = K,
                            nVec = nVec,
                            lambda1 = Float64(lambda[v]),
                            lambda2 = Float64(0),
                            p = p,
                            maxIter = localIter[v])
        end

        if scale
            βmat[:,:,v] = beta ./ sdMat; # rescale by sd
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
