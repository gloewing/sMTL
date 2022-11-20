#' maxEigen: maximum eigenvalue wrapper for Julia TSVD package
#'
#' @param X A matrix.
#' @param intercept A boolean.
#' @param path A string.
#' @return A scalar.
#' @examples
#' set.seed(1)
#'
#' ##########################
#' ##### Simulate Data ######
#' ##########################
#'
#' # create training dataset with 2 covariates
#' X <- matrix(rnorm(2000), ncol = 2)
#'
#' e <- maxEigen(X = as.matrix( X, intercept = FALSE), path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin)"
#' @export


maxEigen = function(X, intercept = TRUE) {
    
    # find julia file
    smtl_path <- .libPaths("sMTL")
    smtl_path <- paste0(smtl_path, "/sMTL/julia/")
        
    maxEgn <- juliaCall("include", paste0(smtl_path, "eigen.jl") ) # max eigenvalue
    
    maxEgn(X = X, intercept = intercept)
    
}
