# finds max eigenvalues for multi study/task or single study
## X: n x p design matrix (feature matrix)
## study: vector of row task or study indices
## intercept: boolean of whether to add a vector of 1s to design matrix


using TSVD

# sparse regression with IHT
function maxEigen(; X::Matrix,
                    study = nothing,
                    intercept = true
                    )

    X = Matrix(X);
    n = size(X, 1);

    if intercept
        # add column of 1s for intercept
        X = hcat(ones(n), X);
    end

    if isnothing(study)
        # if single study
        # Lipschitz constant
        L = tsvd(X)[2][1]; # maximum singular value

    else
        # if multi-study
        study = Int.(study);
        K = length( unique(study) ); # number of studies
        L = 0

        # Lipschitz constant
        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            maxEigen = tsvd(X[indx,:])[2][1]; # max eigenvalue of X^T X
            if (maxEigen > L)
                L = maxEigen
            end
        end

    end

    return L;

end
