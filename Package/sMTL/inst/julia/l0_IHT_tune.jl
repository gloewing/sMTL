# model fitting of sparse regression (single task/dataset)
## X: n x p design matrix (feature matrix)
## y: n x 1 outcome matrix
## rho: Sparsity level (integer)
## beta: p*K initial solution
## scale: whether to center scale features
## lambda>=0: the ridge coefficient
## maxIter: number of max coordinate descent iterations
## localIter: max number of local search iterations
## eig: max eigenvalue for Lipschitz constant

using TSVD, Statistics

include("l0_IHT_opt.jl")
include("l0_IHT_cvx_opt.jl") # convex
include("SingleLS.jl") # local search

# sparse regression with IHT
function L0_iht(; X::Array{Float64,2},
                    y::Array{Float64,1},
                    rho::Integer,
                    beta::Array = 0,
                    scale::Bool = true,
                    lambda = 0,
                    maxIter::Integer = 1000,
                    localIter = 50,
                    eig = nothing
                    )::Array


    n, p = size(X); # number of covaraites
    beta = Array(beta[:,1]); # initial value

    # scale covariates
    if scale
        # scale covariates like glmnet
        Xsd = std(X, dims=1) .* (n - 1) / n; # glmnet style MLE of sd
        X = X ./ Xsd;
        Xsd = Xsd'; # transpose
        Xsd = Xsd[:,1]; # reformat dimensions

        Xsd = vcat(1, Xsd); # add row of ones so standardize intercept by one
        beta .= beta .* Xsd; # rescale warm starts (if not scaled Xsd is a vector of ones)

    end

    # add column of 1s for intercept
    X = hcat(ones(n), X);

    # Lipschitz constant
    if isnothing(eig)
        a2 = tsvd(X)[2][1]; # maximum singular value
    else
        a2 = Float64(eig)
    end

    L = a2.^2 / n # Lipschitz constant without lambda which is added in opt function
    ##############################

    # optimization
    vals = length(lambda)
    βmat = zeros(p + 1, vals) # store all of them -- second index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    if rho < p
        ################
        # sparse setting
        ################

        for v = 1:vals
            # use warm starts as previous value
            beta = L0_iht_opt(X = X,
                                    y = y,
                                    rho = rho,
                                    beta = beta,
                                    L = L,
                                    n = n,
                                    maxIter = maxIter,
                                    lambda = Float64(lambda[v]),
                                    p = p
                                    )

            ###############
            # local search
            ###############
            if localIter[v] > 0
                # run local search if positive number of local search iterations for this lambda1/lambda2 value
                beta = SingleLS(X = X,
                                    y = y,
                                    s = rho,
                                    B = beta,
                                    lambda = Float64(lambda[v]),
                                    n = n,
                                    p = p,
                                    maxIter = localIter[v]
                                    )
            end

            if scale
                βmat[:,v] = beta ./ Xsd; # rescale by sd
            else
                βmat[:,v] = beta
            end
        end

    else
        ################
        # convex setting
        ################

        for v = 1:vals
            # use warm starts as previous value
            beta = L0_iht_cvx_opt(X = X,
                                    y = y,
                                    beta = beta,
                                    L = L,
                                    n = n,
                                    maxIter = maxIter,
                                    lambda = Float64(lambda[v]),
                                    p = p
                                    )

            if scale
                βmat[:,v] = beta ./ Xsd; # rescale by sd
            else
                βmat[:,v] = beta
            end
        end


    end

    if vals == 1
        # if only one tuning value, just return a vector
        return βmat[:,1];
    else
        return βmat;
    end


end
