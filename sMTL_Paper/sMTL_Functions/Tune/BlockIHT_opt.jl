
# sparse regression with IHT -- Block IHT
function BlockIHT_opt(; X::Matrix,
                    y,
                    rho,
                    indxList,
                    beta,
                    K,
                    L,
                    lambda,
                    p,
                    nVec,
                    maxIter = 5000
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # maxIter is maximum number of iterations

    L = L + lambda # update Lipschitz
    ncol = p + 1

    # initialize at 0
    βhat = zeros(ncol, K)
    βprev = zeros(ncol, K)
    b = zeros(ncol, K)

    # initial solution
    βhat = beta
    b = beta; # current solution β
    beta = 0; # delete to save memory

    r = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
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
        βprev = copy(βhat) # previous
        f = 0

        for k = 1:K

            # residual for kth study
            r[k] = X[ indxList[k], :] * b[:,k] - y[ indxList[k] ]

            # gradient for kth study
            g[:, k] = (1 / nVec[k]) * X[ indxList[k], :]' * r[k]
            g[2:end, k] = g[2:end, k] + (lambda) * b[2:end, k] # do not penalize intercept

            # objective for kth study
            f = f + r[k]' * r[k] / (2 * nVec[k]) +
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

      βhat = zeros(ncol, K)
      βhat[idx .+ 1, :] = temp[idx .+ 1, :] # update all except intercept (zero out all except rho biggest)
      βhat[1,:] = temp[1, :]; # update intercept

      # FISTA update
      t = (  1 + sqrt(1 + 4 * t^2)  ) / 2;
      b = βhat + ( (tPrev - 1) / t) * (βhat - βprev);

    end
#####################################

    return βhat;

end
