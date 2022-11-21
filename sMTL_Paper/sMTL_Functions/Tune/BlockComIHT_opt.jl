# Optimization Code
## Block IHT for the common support problem with strength sharing (problem (3) in the write-up)
## b: n*K observation matrix
## A: n*p*K data tensor
## s: Sparsity level (integer)
## x0: p*K initial solution
## lambda1>=0 the ridge coefficient
## lambda2>=0 the strength sharing coefficient

# sparse regression with IHT
function BlockComIHT_opt(; X::Matrix,
                    y,
                    rho,
                    indxList,
                    βhat,
                    K,
                    L,
                    nVec,
                    lambda1,
                    lambda2,
                    p,
                    maxIter = 5000
                    )

    L = L + lambda1 + lambda2
    ncol = p + 1

    # initialize
    #βhat = zeros(ncol, K) # initialize at 0
    #βhat = beta; # current solution β
    #beta = 0; # delete to save memory
    r = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    g = zeros(ncol, K);

    t = 1;
    obj = 1e20;
    iter = 1;
    obj0 = 1e-10;
    bbar = zeros(p)

    ##################################
    # L0 LHT with Ridge penalty
    ##################################

    while (iter <= maxIter)
        objPrev = obj
        # βprev = copy(βhat) # previous
        obj = 0
        bbar = sum(βhat[2:end, :], dims=2 ) / K
        bbar = bbar[:]

        for k = 1:K

            # residual for kth study
            r[k] = X[ indxList[k], :] * βhat[:,k] - y[ indxList[k] ]

            # gradient for kth study
            g[:, k] = (1 / nVec[k]) * X[ indxList[k], :]' * r[k] # update for whole vector
            g[2:end, k] = g[2:end, k] +                          # update just for non-intercept
                                (lambda1) * βhat[2:end, k] +
                                (lambda2) * (βhat[2:end, k] - bbar)

            # objective for kth study
            obj = obj + r[k]' * r[k] / (2 * nVec[k]) +
                        lambda1 / (2) * βhat[2:end, k]' * βhat[2:end, k] +
                        lambda2 / (2) * (βhat[2:end, k] - bbar)' * (βhat[2:end, k] - bbar)
        end

      if ( abs(obj - objPrev)/objPrev < 1e-4 && iter > 10 ) # originally ()/obj0
          break
      end

      if( iter==2)
          obj0 = obj
      end

      iter = iter + 1
      temp = βhat - (1/L) * g # gradient step
      temp_summed = sum( temp.^2,
                        dims = 2) # L2 norm of coefficients summed across studies
      temp_summed = temp_summed[:]

      # projection step
      idx = partialsortperm(temp_summed[2:end], # do not include intercept
                            1:rho,
                            by=abs,
                            rev=true)

      βhat = zeros(ncol, K)
      βhat[idx .+ 1, :] = temp[idx .+ 1, :] # update all except intercept (zero out all except rho biggest)
      βhat[1,:] = temp[1, :]; # update intercept

    end
#####################################

    return βhat;

end

#
# using CSV, DataFrames
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
# X = Matrix(dat[:,3:end]);
# y = (dat[:,2]);
# fit = BlockComIHT(X = X,
#         y = y,
#         study = dat[:,1],
#                     beta =  ones(51, 2),#beta;#
#                     rho = 9,
#                     lambda1 = 0.3,
#                     lambda2 = 0.2,
#                     scale = false,
#                     eig = nothing)
