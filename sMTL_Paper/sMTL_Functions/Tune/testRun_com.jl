include("BlockComIHT_tune.jl")
include("objFun.jl") # local search

# read in data
dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
X = Matrix(dat[:,3:end]); # design matrix
y = (dat[:,2]); # outcome
study = dat[:,1] # vector of study indices for each observation
K = length(unique(study))

# now run a path of IHT (each potentially with a local search after each IHT)
# the path fits at each of the following lambda values
# keep these lambda fixed for the experiment
lambda1 =  ones(4) # vector of lambda1
lambda2 = ones(4) #ones(4)

fit = BlockComIHT(X = X,
        y = y,
        study = study,
                    beta =  zeros(51, 2), # initialize
                    rho = 8, # number of non-zero coefs ignoring intercept
                    lambda1 = lambda1,
                    lambda2 = lambda2,
                    maxIter = 10000, # number of IHT iterations
                    scale = true, # scale covariates before fitting
                    eig = nothing, # do SVD inside function
                    localIter = [0 0 0 10] # number of local search iterations at each point in path--
                    )


# check objective value for each solution in the path for the [itr] solution
# if you check for itr = 4, that is the only one that has local search, and it is larger than itr = 3 (no local search)
itr = 4
objFun( X = X,
        y = y,
        study = study,
                    beta = fit[:,:, itr],
                    lambda1 = lambda1[ itr ],
                    lambda2 = lambda2[ itr ],
                    lambda_z = 0,
                    )

# number of non-zeros per study (not including intercept)
size(findall(x -> x.> 1e-9, abs.(fit[2:end, :,1])))[1] / K

#
# include("l0_IHT_tune.jl")
#
# lambda1 = ones(4) * 3
#
# fit1 = L0_iht(X = X,
#                     y = y,
#                     rho = 10,
#                     beta = zeros(size(X,2) + 1),
#                     scale = true,
#                     lambda = lambda1,
#                     localIter = [ 0 0 0 20]
# );
#
# itr = 4
# objFun( X = X,
#         y = y,
#         study = ones(size(X)[1]),
#                     beta = fit1[:, itr],
#                     lambda1 = lambda1[ itr ],
#                     lambda2 = 0, #lambda2[ itr ],
#                     lambda_z = 0,
#                     )
#
# size(findall(x -> x.> 1e-9, abs.(fit[2:end, :,1])))[1] / K
