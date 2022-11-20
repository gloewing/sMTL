
# sparse regression with IHT
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

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations

    L = L + lambda;
    ncol = p + 1

    # initialize
    βprev = zeros(p); # previous β
    βhat = copy(beta); # current solution β
    # b = beta; # update

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

# ###############################
# using CSV, Random, DataFrames, Statistics
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research Final/Mas-o-Menos/dat", DataFrame);
# X = Matrix(dat[:,2:end]);
# y = (dat[:,1]);
#
# fit1 = L0_iht(X = X,
#                     y = y,
#                     rho = 5,
#                     beta = ones(size(X,2) + 1),
#                     scale = true,
#                     lambda = 0.1
#                     );
