#' maxEigen: maximum eigenvalue wrapper for Julia TSVD package. internal package use only
#'
#' @param X A matrix.
#' @param intercept A boolean.
#' @return A scalar.
#' @examples
#'
#' ##########################
#' ##### Simulate Data ######
#' ##########################
#'
#' # create training dataset with 2 covariates
#' X <- matrix(rnorm(2000), ncol = 2)
#'
#' \dontrun{
#' e <- maxEigen(X = as.matrix( X, intercept = FALSE), 
#'      path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin)" }
#'      
#' @import JuliaConnectoR
#' @export


maxEigen = function(X, intercept = TRUE) {
    
    # find julia file
    smtl_path <- .libPaths("sMTL")
    smtl_path <- paste0(smtl_path, "/sMTL/julia/")
        
    maxEgn <- juliaCall("include", paste0(smtl_path, "eigenFn.jl") ) # max eigenvalue
    
    maxEgn(X = X, intercept = intercept)
    
}
