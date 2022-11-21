# no random restarts--initialize with IHT solution

# MoM L0 with IHT
function MoM_iht_opt(; X::Matrix,
                    y,
                    rho,
                    L,
                    d,
                    n,
                    p,
                    beta,
                    maxIter = 5000
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations

    # initialize
    ncol = p + 1

    # initialize α
    indx = findall(x -> x > 0, beta); # non-zero betas
    α_min = minimum( abs.(beta[indx]) ) # lowest non-zero beta
    α_max = maximum( abs.(beta[indx]) ) # highest non-zero beta
    h = (α_max - α_min) / (d + 1);
    α = collect(α_min:h:α_max)[2:(end-1)];

    #βprev = zeros(ncol); # previous β
    βhat = beta; # current solution β
    #b = beta; # update

    t = 1;
    obj = 1e20;
    iter = 1;
    obj0 = 1e-10;

    ##################################
    # L0 LHT with Ridge penalty
    ##################################
    while (iter <= maxIter)
      objPrev = obj;
      #tPrev = t;
      #βprev = copy(βhat); # previous value

      r = X * βhat - y; # residual
      g = (1 / n) .* X' * r; # gradient  #
      obj = (r' * r) / (2 * n); # objective (includes ridge penalty) #

      # test for convergence
      if (  abs(obj - objPrev) / objPrev < 1e-5 && iter > 3  )
          # return βhat # removed this return
          break
      end

      if ( iter == 2 )
          obj0 = obj;
      end

      iter = iter + 1; # iteration
      temp = βhat - (1/L) * g; # gradient step without FISTA scaling

      # projection step: zero-out all but top rho highest entries (except intercept) and set rest to sign(β)
      idx = partialsortperm( temp[2:end],
                            1:rho,
                            by = abs,
                            rev = true); # indices of largest βhat entries (except intercept)

      # update βhat
      βhat[2:end] = zeros(p);
      βhat[1] = temp[1]; # update intercept
      #βhat[idx .+ 1] = temp[idx .+ 1]; # add one so indices account for intercept
      distMat = (temp[idx .+ 1] .- α').^2; # distance oh non-zero betas from current alphas
      minIndx = map(argmin, distMat[i,:] for i = 1:size(distMat, 1) ); # indices of closest values

      for j = 1:d
          indxD = findall(x -> x == j, minIndx ); # indices of betas in jth cluster
          indx2 = (idx .+ 1)[indxD] # map to indices of original beta vector
          α[j] = mean( abs.(temp[indx2]) ); # average of the magnitudes
          βhat[indx2] = sign.( temp[indx2] ) .* α[j] # sign times the mean of the betas for jth cluster
      end

      # βhat[idx .+ 1] = sign.(temp[idx .+ 1]); # add one so indices account for intercept

      # update α
      # used: https://en.wikipedia.org/wiki/Simple_linear_regression (without intercept where ỹ = y - β_0 )
      # α = (  (y .- βhat[1])' * (X[:,2:end] *  βhat[2:end])  )  / sum( (X[:, 2:end] *  βhat[2:end]).^2 )

      # # update βhat
      # βhat[2:end] = α .* βhat[2:end]

      # FISTA update
      # t = (  1 + sqrt(1 + 4 * t^2)  ) / 2;
      # b = βhat + ( (tPrev - 1) / t) * (βhat - βprev);
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
