# Optimization Code
## X: n x p design matrix (feature matrix)
## y: n x 1 outcome vector
## rho: Sparsity level (integer)
## idxList: list of task row indices
## βhat: p*K initial solution
## K: number of tasks
## L: Lipschitz constant
## n: sample size
## lambda1>=0: the ridge coefficient
## lambda2>=0: the coefficient value strength sharing coefficient (Bbar penalty)
## p: num covariates/features
## maxIter: number of max coordinate descent iterations

# sparse regression with IHT
function BlockComIHT_opt(; X::Array{Float64,2},
                    y::Array{Float64,1},
                    rho::Integer,
                    indxList::Array{Array{Int64,1}},
                    βhat::Array{Float64,2},
                    K::Integer,
                    L::Float64,
                    nVec::Array{Int64,1},
                    lambda1::Float64,
                    lambda2::Float64,
                    p::Integer,
                    maxIter::Integer = 5000
                    )::Array{Float64,2}

    L = L + lambda1 + lambda2
    ncol = p + 1

    # initialize
    r = [ zeros( nVec[i] ) for i in 1:K ];
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

      if ( abs(obj - objPrev)/objPrev < 1e-4 && iter > 10 )
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
