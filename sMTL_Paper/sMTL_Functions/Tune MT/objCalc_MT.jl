## objective

function objCalc(; X,
                    y,
                    study = nothing, # dummy
                    beta = 0,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0
                    )

                    y = Matrix(y);
                    X = Matrix(X);
                    n, p = size(X); # number of covaraites
                    X = hcat(ones(n), X);
                    K = size(y, 2); # number of tasks

                    objVec = zeros( length(lambda1) )

                    for j = 1:length(lambda1)
                        # iterate across tuning values

                        obj = 0

                        if K > 1
                            βhat = Matrix( beta[:,:,j] ); # initial value

                            bbar = sum(βhat[2:end, :], dims=2 ) / K
                            bbar = bbar[:]

                            z = zeros(p, K);
                            idx1 = findall(x -> x.> 1e-9, abs.(βhat[2:end, :]))
                            z[idx1] = ones(size(idx1))
                            z_bar = sum(z, dims=2) / K
                            z_bar = z_bar[:]

                            for k = 1:K

                                nk = n # sample size

                                # residual for kth study
                                r = X * βhat[:,k] - y[ :, k ]

                                # objective for kth study
                                obj = obj + r' * r / (2 * nk) +
                                            lambda1[j] / 2 * βhat[2:end, k]' * βhat[2:end, k] +
                                            lambda2[j] / 2 * (βhat[2:end, k] - bbar)' * (βhat[2:end, k] - bbar) +
                                            lambda_z[j] / 2 * (z[:,k] - z_bar)' * (z[:,k] - z_bar)


                            end

                        else
                            # if only one study

                            βhat = beta[:]; # initial value

                                nk = n
                                # residual for kth study
                                r = X * βhat - y

                                # objective for kth study
                                obj = obj + r' * r / (2 * nk) +
                                            lambda1[j] / 2 * βhat[2:end]' * βhat[2:end]

                        end

                        objVec[j] = obj
                    end




                    return objVec
end
