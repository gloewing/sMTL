using TSVD, Statistics

include("l0_IHT_opt.jl")
include("SingleLS.jl") # local search

# sparse regression with IHT
function L0_iht(; X,
                    y,
                    rho,
                    beta = 0,
                    scale = true,
                    lambda = 0,
                    maxIter = 1000,
                    localIter = 50,
                    eig = nothing
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations

    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Array(beta[:,1]); # initial value
    y = Array(y);
    rho = Int64(rho);

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

    for v = 1:vals
        # use warm starts as previous value
        beta = L0_iht_opt(X = X,
                                y = y,
                                rho = rho,
                                beta = beta,
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
            beta = SingleLS(X = X,
                                y = y,
                                rho = rho,
                                beta = beta,
                                lambda = lambda[v],
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

    if vals == 1
        # if only one tuning value, just return a vector
        return βmat[:,1];
    else
        return βmat;
    end


end
