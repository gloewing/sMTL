# Model Fitting Code
## X: n x p design matrix (feature matrix)
## y: n x K outcome matrix
## rho: Sparsity level (integer)
## study: study/task row indices
## beta: p*K initial solution
## scale: whether to center scale features
## idxList: list of task row indices
## eig: max eigenvalue of all design matrices
## idx: active set indices
## lambda1>=0: the ridge coefficient
## lambda2>=0: the coefficient value strength sharing coefficient (Bbar penalty)
## lambda_z>=0: the coefficient support strength sharing coefficient (Zbar penalty)
## p: number of features/covariates
## maxIter: max iterations of coordiante descent
## localIter: local search iterations
## maxIter_in: number of inner coordinate descent iterations (within active sets)
## maxIter_out: number of outer coordinate descent iterations
## maxIter: max iterations of coordinate descent
## WSmethod: dummy variable
## ASpass: dummy variable
## ASmultiplier: dummy variable
## svdFlag: dummy variable

using TSVD, Statistics

include("BlockComIHT_inexact_diffAS_optTest_MT.jl") # IHT algorithm
include("BlockComIHT_inexact_diffAS_opt_oldTest_MT.jl") # IHT algorithm but active set constructed inside IHT
include("BlockInexactLS_MT.jl") # local search algorithm
include("l0_IHT_opt.jl") # individual L0 regressions to find active set

# sparse regression with IHT and local search
function BlockComIHT_inexactAS_diff_MT(; X::Array{Float64,2},
                    y::Array{Float64,2},
                    rho::Integer,
                    study = nothing,
                    beta::Array{Float64,2} = 0,
                    scale::Bool = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0,
                    maxIter::Integer = 5000,
                    localIter = 50,
                    idx = nothing,
                    maxIter_in = nothing,
                    maxIter_out = nothing,
                    eig = nothing,
                    eigenVec = nothing, # dummy variable
                    WSmethod::Integer = 2, # dummy variable
                    ASpass::Bool = true, # dummy variable
                    ASmultiplier::Integer = 4,
                    svdFlag::Bool = false,
                    )::Array


    n, p = size(X); # number of covaraites
    K = size(y, 2);  # number of studies

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

        sdMat = hcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat'; # current solution β

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)

        if svdFlag
            _, singVals, _ = svd( X, alg = LinearAlgebra.QRIteration() )
            eig = singVals[1]
        else
            eig = tsvd( X )[2][1]; # max eigenvalue of X^T X
        end

    else
        eig = Float64(eig)
    end


    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    ####################################################################
    # find initial active set with individual L0 regressions
    ####################################################################
    if isnothing(idx)

        rhoStar = min(rho * ASmultiplier, p); # active set set is bigger than actual rho
        # if no active set inidices provided, use individual sparse regressions to get an initial active set
        idx = [ Vector{Int64}(undef, rhoStar) for i in 1:K]; # list of vectors of indices of studies#zeros(p, K)

        for i = 1:K
            # this alters
            beta[:,i] = L0_iht_opt(X = X,
                                    y = y[:, i ],
                                    rho = rhoStar,
                                    beta = beta[:,i],
                                    L = eig^2 / n,
                                    n = n,
                                    maxIter = maxIter,
                                    lambda = Float64(lambda1[1]), # use first ridge term
                                    p = p
                                    )

            idx[i] = findall(x-> x.>1e-9, abs.( beta[2:end, i] ) ) # save non-zero elements (not including intercept)
        end
    end

    ###############
    # optimization
    ###############

    for v = 1:vals

        if WSmethod == 1
            # use warm starts as previous value
            beta, idx1 = BlockComIHT_inexactAS_diff_opt_MT(X = X,
                                            y = y,
                                            rho = rho,
                                            B = beta,
                                            K = K,
                                            eig = eig,
                                            n = n,
                                            maxIter_in = maxIter_in,
                                            maxIter_out = maxIter_out,
                                            lambda1 = Float64(lambda1[v]),
                                            lambda2 = Float64(lambda2[v]),
                                            lambda_z = Float64(lambda_z[v]),
                                            idx = idx,
                                            p = p
                                            );
            if ASpass
                idx = idx1;
            end

        else

            beta = BlockComIHT_inexactAS_diff_old_opt_MT(X = X,
                                            y = y,
                                            rho = rho,
                                            B = beta,
                                            K = K,
                                            eig = eig,
                                            n = n,
                                            maxIter_in = maxIter_in,
                                            maxIter_out = maxIter_out,
                                            lambda1 = Float64(lambda1[v]),
                                            lambda2 = Float64(lambda2[v]),
                                            lambda_z = Float64(lambda_z[v]),
                                            p = p
                                            );
        end

        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockInexactLS_MT(X = X,
                                y = y,
                                s = rho,
                                beta = beta,
                                lambda1 = lambda1[v],
                                lambda2 = lambda2[v],
                                lambda_z = lambda_z[v],
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
