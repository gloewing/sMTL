
using TSVD, Statistics
include("BlockIHT_opt.jl")
include("BlockLS.jl") # local search algorithm -- same as with lambda_z but set to 0 always

# sparse regression with IHT -- Block IHT
function BlockIHT(; X,
                    y,
                    rho,
                    study,
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

    y = Array(y);
    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Matrix(beta); # initial value
    rho = Int64(rho);
    study = Int.(study);
    K = length( unique(study) ); # number of studies
    indxList = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    nVec = zeros(K); # vector of sample sizes of studies

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
        # sdMat = ones(p, K); # K x p matrix to save std of covariates of each study

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
                                        lambda = lambda[v],
                                        p = p
        )

        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockLS(X = X,
                            y = y,
                            rho = rho,
                            indxList = indxList,
                            B = beta,
                            K = K,
                            nVec = nVec,
                            lambda1 = lambda[v],
                            lambda2 = 0,
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
