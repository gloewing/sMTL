using TSVD

## Block ccoridnate descent for z-z_bar problem
## y: n*K observation matrix
## X: n*p*K data tensor
## s: Sparsity level (integer)
## B0: p*K initial solution
## lambda1>=0 the ridge coefficient
## lambda2>=0 the strength sharing coefficient
## lambda_z>=0 the z regularization parameter
## idx: indices of initial active set

function BlockInexactActiveSet(y, X, s, B0, lambda1, lambda2, lambda_z, idx)

    n,p, K = size(X)
    L = 0
    for k = 1:K
        a1,a2,a3 = tsvd(X[:,:,k])
        if (a2[1] > L)
            L = a2[1]
        end
    end

    L = L^2*sqrt(K) + lambda1 + lambda2


    B = B0
    z = zeros(p,K)
    B_bar = zeros(p)
    z_bar = zeros(p)
    Z_bar = zeros(p,K)

    obj = 1e20
    iter_in = 1
    iter_out = 1
    maxIter_in = 1000
    maxIter_out = 1000
    r = zeros(n, K)
    g = zeros(p, K)






    while (iter_out <= maxIter_out)

        objPrev = obj
        BPrev = copy(B)




        L_active = 0
        for k = 1:K
            a1,a2,a3 = tsvd(X[:,idx,k])
            if (a2[1] > L_active)
                L_active = a2[1]
            end
        end
        L_active = L_active^2*sqrt(K) + lambda1 + lambda2
        B_bar_active = zeros(length(idx))
        z_bar_active = zeros(length(idx))
        r_active = zeros(n,K)
        g_active = zeros(length(idx),K)
        B_active = B[idx,:]
        z_active = z[idx, :]
        while (iter_in <= maxIter_in)


            objPrev = obj
            B_bar_active = sum(B_active, dims=2)/K
            z_bar_active = sum(z_active, dims=2)/K
            z_bar_active = z_bar_active[:]
            B_bar_active = B_bar_active[:]
            f = 0
            for k = 1:K
                r_active[:, k] = X[:,idx,k]*B_active[:,k] - y[:,k]
                g_active[:, k] = X[:,idx,k]'*r_active[:,k] + lambda1*B_active[:,k] + lambda2*(B_active[:,k] - B_bar_active)
                f = f + r_active[:,k]'*r_active[:,k]/2 + lambda1*B_active[:,k]'*B_active[:,k]/2 + lambda2*(B_active[:,k] - B_bar_active)'*(B_active[:,k] - B_bar_active)/2 + lambda_z*(z[idx,k] - z_bar_active)'*(z[idx,k] - z_bar_active)/2
            end
            obj = f

            if (abs(obj - objPrev)/objPrev < 1e-6)
                println(obj)
                break
            end
            iter_in = iter_in + 1
            Z_bar_active = z_bar_active*ones(K)'
            B_temp_active = B_active - 0.5*(1/L)*g_active
            z_active = zeros(length(idx), K)
            idx_active = findall(x-> x.>1e-9, abs.(B_temp_active))
            z_active[idx_active] = ones(size(idx_active))
            cost_active =  lambda_z*Z_bar_active.^2/L_active + B_temp_active.^2 - lambda_z*(z_active-Z_bar_active).^2/L_active
            z_active = zeros(length(idx), K)
            for k=1:K
                idx1 = partialsortperm(cost_active[:,k], 1:s, rev=true)
                z_active[idx1,k] = ones(size(idx1))
            end


            B_active = B_temp_active.*z_active
            B[idx,:] = B_active[:,:]
            z[idx,:] = z_active[:,:]
        end







        B_bar = sum(B, dims=2)/K
        z_bar = sum(z, dims=2)/K
        z_bar = z_bar[:]
        B_bar = B_bar[:]
        f = 0
        for k = 1:K
            r[:, k] = X[:,:,k]*B[:,k] - y[:,k]
            g[:, k] = X[:,:,k]'*r[:,k] + lambda1*B[:,k] + lambda2*(B[:,k] - B_bar)
            f = f + r[:,k]'*r[:,k]/2 + lambda1*B[:,k]'*B[:,k]/2 + lambda2*(B[:,k] - B_bar)'*(B[:,k] - B_bar)/2 + lambda_z*(z[:,k] - z_bar)'*(z[:,k] - z_bar)/2
        end
      obj = f


      iter_out = iter_out + 1
      Z_bar = z_bar*ones(K)'
      B_temp = B - 0.5*(1/L)*g

      z = zeros(p, K)
      idx1 = findall(x-> x.>1e-9, abs.(B_temp))
      z[idx1] = ones(size(idx1))
      cost =  lambda_z*Z_bar.^2/L + B_temp.^2 - lambda_z*(z-Z_bar).^2/L
      z = zeros(p, K)
      flag = 0
      for k=1:K
          idx1 = partialsortperm(cost[:,k], 1:s, rev=true)
          z[idx1,k] = ones(size(idx1))
          for jj = 1:length(idx1)
              if (sum(idx .== idx1[jj]) == 0)
                  idx = [idx; idx1[jj]]
                  flag = 1
              end
          end
      end
     B = B_temp.*z
     if (flag == 0)
         break
     end


    end
println(obj)
return B

end

n = 100
p = 10
K = 3

y = Matrix(randn(n, K))
X = (randn(n, p, K))
s = 4
B0 = randn(p, K)
lambda1 = 0
lambda2 = 0
lambda_z = 0
idx = [1 2 3 4]
idx = idx[:]


fit = BlockInexactActiveSet(y, X, s, B0, lambda1, lambda2, lambda_z, idx)




dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
X = Matrix(dat[:,3:end]);
y = (dat[:,2]);
p = size(X)[2]
n = 100
K = 2

y2 = Matrix(randn(n, K))
X2 = (randn(n, p, K))
B0 = randn(p, K)

X2[:,:,1] = X[1:100,:]
X2[:,:,2] = X[101:200,:]
y2[:,1] = y[1:100]
y2[:,2] = y[101:200]
lambda1 = 1
lambda2 = 1
lambda_z = 0.2
idx = idx[:]

B0 = randn(p, K)

fit = BlockInexactActiveSet(y2, X2, s, B0, lambda1, lambda2, lambda_z, idx)
print(fit)
