using TSVD, Statistics # MatrixImpute, Ipopt, DataFrames,

include("MoM_IHT_opt.jl")
# include("MoM_IHT_opt_fista.jl")
include("l0_IHT_opt.jl") # for warm starts

# MoM L0 IHT
function MoM_iht(; X,
                    y,
                    rho,
                    multiplier = 3,
                    scale = false,
                    maxIter = 5000,
                    warmStartIter = 50,
                    eig = nothing,
                    itrs = 10
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # itrs is number of random restarts -- no longer used but kept for ease
    # warmStartIter is number of L0 IHT iterations to use before MoM

    X = Matrix(X);
    n, p = size(X); # number of covaraites
    y = Array(y);
    #lambda = Array(lambda);
    rho = Int64.(rho);
    itrs = Int(itrs);
    beta = zeros(p + 1) # initialize for IHT L0 Regression (which is warm start for MoM)

    # scale covariates
    if scale
        # scale covariates like glmnet
        Xsd = std(X, dims=1) .* (n - 1) / n; # glmnet style MLE of sd
        X = X ./ Xsd;
        Xsd = Xsd'; # transpose
        Xsd = Xsd[:,1]; # reformat dimensions
        # Ysd = std(y) * (n - 1) / n; # glmnet style MLE of sd
        # lambda = lambda / Ysd; # scale tuning parameter by std of y

        Xsd = vcat(1, Xsd); # add row of ones so standardize intercept by one
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
    vals = length(rho)
    βmat = zeros(p + 1, vals) # store all of them -- second index is tuning value
    warmStartFlag = true # need warm start
    rhoWS = 0 # use larger value of rho (e.g., rho * 4) for warm start IHT

    for v = 1:vals

        # determine whether warm start is needed and set rhoWS
        if 2 * rho[v] <= rhoWS <= multiplier * rho[v]
            warmStartFlag = false # if rhoWS is 2-3 times as large as current rho, no need to run warm start again
        else
            rhoWS = min(rho[v] * multiplier, p)
            warmStartFlag = true # run IHT for warm start
        end

        if warmStartFlag
            # iht regression warm start
            global betaStart = L0_iht_opt(X = X,
                                    y = y,
                                    rho = rhoWS,
                                    beta = beta,
                                    L = L,
                                    n = n,
                                    maxIter = warmStartIter, # does not need to be at optimum
                                    lambda = 0,
                                    p = p
                                    )

        end

        βmat[:,v] = MoM_iht_opt(X = X,
                                y = y,
                                rho = rho[v],
                                L = L,
                                n = n,
                                beta = betaStart,
                                maxIter = maxIter,
                                p = p
                                )

        if scale
            βmat[:,v] = βmat[:,v] ./ Xsd; # rescale by sd
        end

    end

    if vals == 1
        # if only one tuning value, just return a vector
        return βmat[:,1];
    else
        return βmat;
    end

end

# ###############################
# # # # # using CSV, Random, DataFrames, Statistics
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research Final/Mas-o-Menos/dat2", DataFrame);
# X = Matrix(dat[:,2:end]);
# y = (dat[:,1]);
#
# fit1 = MoM_iht(X = X,
#                     y = y,
#                     rho = [2 3 4],
#                     #beta = ones(size(X,2) + 1),
#                     scale = false,
#                     itrs = 10
#                     );
