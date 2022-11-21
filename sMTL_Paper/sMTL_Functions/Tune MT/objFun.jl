## objective

function objFun(; X,
                    y,
                    study,
                    beta = 0,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0
                    )

                    y = Array(y);
                    X = Matrix(X);
                    n, p = size(X); # number of covaraites
                    X = hcat(ones(n), X);
                    study = Int.(study);
                    K = length( unique(study) ); # number of studies

                    obj = 0

                    if K > 1
                        βhat = Matrix(beta); # initial value

                        bbar = sum(βhat[2:end, :], dims=2 ) / K
                        bbar = bbar[:]

                        z = zeros(p, K);
                        idx1 = findall(x -> x.> 1e-9, abs.(βhat[2:end, :]))
                        z[idx1] = ones(size(idx1))
                        z_bar = sum(z, dims=2) / K
                        z_bar = z_bar[:]

                        for k = 1:K

                            indx = findall(x -> x == k, study); # indices of rows for ith study
                            nk = length(indx) # sample size

                            # residual for kth study
                            r = X[ indx, :] * βhat[:,k] - y[ indx ]

                            # objective for kth study
                            obj = obj + r' * r / (2 * nk) +
                                        lambda1 / 2 * βhat[2:end, k]' * βhat[2:end, k] +
                                        lambda2 / 2 * (βhat[2:end, k] - bbar)' * (βhat[2:end, k] - bbar) +
                                        lambda_z / 2 * (z[:,k] - z_bar)' * (z[:,k] - z_bar)


                        end

                    else
                        # if only one study

                        βhat = beta[:]; # initial value

                            nk = n
                            # residual for kth study
                            r = X * βhat - y

                            # objective for kth study
                            obj = obj + r' * r / (2 * nk) +
                                        lambda1 / 2 * βhat[2:end]' * βhat[2:end]

                    end



                    return obj
end
