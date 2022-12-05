# Optimization Code
# sparse regression with IHT
function BlockComIHT_inexact_opt_MT(; X::Matrix,
                    y,
                    s,
                    #indxList,
                    B,
                    K,
                    L,
                    n,
                    #nVec,
                    lambda1,
                    lambda2,
                    lambda_z,
                    p,
                    maxIter = 5000
                    )

    L = L + lambda1 + lambda2
    ncol = p + 1

    # initialize
    z = zeros(p, K)
    B_bar = zeros(ncol)
    z_bar = zeros(p)
    Z_bar = zeros(p, K)

    r = zeros(n, K)  # list of vectors of indices of studies
    g = zeros(ncol, K);

    obj = 1e20;
    iter = 1;

    cost = zeros(p, K)
    idx = findall(x-> x.>1e-9, abs.(B[2:end,:]) ) # do not calculate for intercept
    z[idx] = ones(size(idx))

    ##################################
    # L0 LHT with Ridge penalty
    ##################################

    while (iter <= maxIter)

        objPrev = obj
        BPrev = copy(B)
        B_bar = sum(B[2:end, :], dims=2) / K
        z_bar = sum(z, dims=2) / K
        z_bar = z_bar[:]
        B_bar = B_bar[:]
        f = 0

        for k = 1:K

            # residual for kth study
            r[:, k] = X * B[:,k] - y[ :, k ]

            # gradient for kth study
            g[:, k] = (1 / n) * X' * r[:, k] # update for whole vector
            g[2:end, k] = g[2:end, k] +                          # update just for non-intercept
                        lambda1 * B[2:end, k] +
                        lambda2 * (B[2:end, k] - B_bar)

            # objective for kth study
            f = f + r[:, k]' * r[:, k] / (2 * n) +
                        lambda1 / 2 * B[2:end, k]' * B[2:end, k] +
                        lambda2 / 2 * (B[2:end, k] - B_bar)' * (B[2:end, k] - B_bar) +
                        lambda_z / 2 * (z[:,k] - z_bar)' * (z[:, k] - z_bar)
        end

      #########################################
      obj = f

      if (abs(obj - objPrev)/objPrev < 1e-5 && iter>20)
          break
      end

      iter = iter + 1
      Z_bar = z_bar * ones(K)'
      B_temp = B - 0.5 * (1/L) * g

      z = zeros(p, K)
      idx = findall(x-> x.>1e-9, abs.(B_temp[2:end,:]) ) # do not do this for intercept
      z[idx] = ones(size(idx))
      # cost =  lambda_z * Z_bar.^2 / L + B_temp[2:end,:].^2 - lambda_z * (z-Z_bar).^2 / L
      cost =  lambda_z / L * (Z_bar.^2 - (z-Z_bar).^2) # scale by nVec (below) since our objective is altered
      z = zeros(p, K)

      for k=1:K
          cost_k = B_temp[2:end, k].^2 + n * cost[:,k] 
          idx = partialsortperm(cost_k, 
                                1:s,
                                rev=true)
          z[idx, k] = ones(size(idx))
      end

     B[2:end,:] = B_temp[2:end,:] .* z
     B[1,:] = B_temp[1,:] # update intercept

    end

    return B

end
