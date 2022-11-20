## y: N x K outcome
## X: N x p design matrix
## rho: Sparsity level (integer)
## beta: p*K initial solution
## lambda>=0 the ridge coefficient


# sparse regression with IHT -- Block IHT
function BlockIHT_opt_MT(; X::Array{Float64,2},
                    y::Array{Float64,2},
                    rho::Integer,
                    beta::Array{Float64,2},
                    K::Integer,
                    L::Float64,
                    lambda::Float64,
                    p::Integer,
                    n::Integer,
                    maxIter::Integer = 5000
                    )::Array{Float64,2}

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # maxIter is maximum number of iterations

    L = L + lambda # update Lipschitz
    ncol = p + 1
    #n = size(X, 1)

    # initialize at 0
    # βhat = zeros(ncol, K)
    βprev = zeros(ncol, K)
    b = zeros(ncol, K)

    # initial solution
    b = copy(beta); # current solution β
    # βhat = beta
    # beta = 0; # delete to save memory

    r = zeros(n, K) #[Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    g = zeros(ncol, K);

    t = 1;
    obj = 1e20;
    iter = 1;
    obj0 = 1e-10;

    ##################################
    # L0 LHT with Ridge penalty
    ##################################

    while (iter <= maxIter)
        objPrev = obj
        tPrev = t
        βprev = copy(beta) # previous
        f = 0

        for k = 1:K

            # residual for kth study
            r[:,k] = X * b[:,k] - y[ :, k ]

            # gradient for kth study
            g[:, k] = (1 / n) * X' * r[:,k]
            g[2:end, k] = g[2:end, k] + (lambda) * b[2:end, k] # do not penalize intercept

            # objective for kth study
            f = f + r[:,k]' * r[:,k] / (2 * n) +
                        lambda / 2 * b[2:end, k]' * b[2:end, k] # do not penalize intercept

        end

        obj = f # update objective value

      if ( abs(obj - objPrev)/objPrev < 1e-4 && iter > 3 ) # originally /obj0
          break
      end

      if( iter==2)
          obj0 = obj
      end

      iter = iter + 1
      temp = b - (1/L) * g # gradient step
      temp_summed = sum( temp.^2,
                        dims = 2)
      temp_summed = temp_summed[:]

      # projection step
      idx = partialsortperm(temp_summed[2:end], # do not include intercept
                            1:rho,
                            by=abs,
                            rev=true)

      beta = zeros(ncol, K)
      beta[idx .+ 1, :] = temp[idx .+ 1, :] # update all except intercept (zero out all except rho biggest)
      beta[1,:] = temp[1, :]; # update intercept

      # FISTA update
      t = (  1 + sqrt(1 + 4 * t^2)  ) / 2;
      b = beta + ( (tPrev - 1) / t) * (beta - βprev);

    end
#####################################

    return beta;

end
