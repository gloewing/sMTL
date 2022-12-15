# sparse regression with IHT
# Optimization Code
## X: n x p design matrix (feature matrix)
## y: n x 1 outcome vector
## rho: sparsity level
## beta: p x K initial solution
## n: sample size
## L: Lipschitz constant
## lambda>=0: the ridge coefficient
## p: num covariates/features
## maxIter: number of max coordinate descent iterations

function L0_iht_opt(; X::Matrix,
                    y::Array,
                    rho::Integer,
                    beta::Array,
                    n::Integer,
                    L::Float64,
                    lambda::Float64,
                    p::Integer,
                    maxIter::Integer = 1000
                    )::Array


    L = L + lambda;
    ncol = p + 1

    # initialize
    βprev = zeros(p); # previous β
    βhat = copy(beta); # current solution β

    t = 1;
    obj = 1e20;
    iter = 1;
    obj0 = 1e-10;

    ##################################
    # L0 LHT with Ridge penalty
    ##################################
    while (iter <= maxIter)
      objPrev = obj;
      tPrev = t;
      βprev = copy(βhat); # previous value

      r = X * beta - y; # residual
      g = (1 / n) .* X' * r # gradient update for whole vector
      g[2:end] = g[2:end] + lambda .* βhat[2:end]; # gradient update for non-intercept terms

      obj = (r' * r) / (2 * n) + (lambda / 2) * (βhat[2:end]' * βhat[2:end]); # objective (includes ridge penalty)

      # test for convergence
      if (  abs(obj - objPrev) / obj0 < 1e-5 && iter > 3  )
          # return βhat # removed this return
          break
      end

      if ( iter == 2 )
          obj0 = obj;
      end

      iter = iter + 1; # iteration
      temp = beta - (1/L) * g; # gradient step without FISTA scaling

      # projection step: zero-out all but top rho highest entries (except intercept)
      idx = partialsortperm( temp[2:end],
                            1:rho,
                            by = abs,
                            rev = true); # indices of largest βhat entries (except intercept)
      βhat[2:end] = zeros(p);
      βhat[idx .+ 1] = temp[idx .+ 1]; # add one so indices account for intercept
      βhat[1] = temp[1]; # update intercept

      # FISTA update
      t = (  1 + sqrt(1 + 4 * t^2)  ) / 2;
      beta = βhat + ( (tPrev - 1) / t) * (βhat - βprev);
    end


    return βhat;

end
