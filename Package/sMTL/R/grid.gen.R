#' grid.gen: generate grid for ross-validation function
#'
#' @param y A numeric vector
#' @param X A matrix
#' @param study An integer vector
#' @param grid A dataframe
#' @param nfold An integer
#' @param commonSupp A boolean
#' @param multiTask A boolean
#' @param maxIter An integer
#' @param LocSrch_skip An integer
#' @param LocSrch_maxIter An integer
#' @param messageInd A boolean
#' @param independent.regs A boolean
#' @return A list
#' @examples
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Installed But Packages NEED INSTALLATION  ######
#' #####################################################################################
#' smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", installJulia = TRUE, installPackages = TRUE)
#' @import JuliaConnectoR
#' @import dplyr
#' @export

grid.gen = function(y, 
                   X, 
                   study = NA, 
                   grid,
                   commonSupp = FALSE,
                   lambda_1 = TRUE,
                   lambda_2 = FALSE,
                   lambda_z = TRUE,
                   multiTask = TRUE # only used if study indices provided, then use this to distinguish between a "hoso" and " multiTask" tuning 
) {
    
    p <- ncol(X)
    
    if( is.matrix(y) ){
        
        # make sure y is a matrix with multiple columns
        if( ncol(y) > 1 ){
            reg_type <- "multiLabel"
            K <- ncol(y)
        }else{
            # if only 1 column, then coerce into vector
            y <- as.numeric(y)
            K <- 1
        }    
        
    }
    
    if( is.vector(y) ){
        
        # y is a vector and no studies given then just L0 problem
        if( anyNA(study) ){
            reg_type <- method <- "L0"
            K <- 1
        }else{
            # if study is given
            study <- as.integer( as.factor(study) ) # make integer for Julia type
            reg_type <- "multiStudy"
            K <- length( unique(study) )
            
        }
    }
    
    # ridge penalty
    if(lambda_1){
        # baseline cardinality set to s = 10 
        s_baseline <- 10
        
        lambda1_vec <- c(1e-7, 1e-5, 1e-3, 1e-1, 1)
        
    }else{
        lambda1_vec <- 0
    }
    
    # bbar penalty
    if(lambda_2){
        lambda2_vec <- 10^(seq(-5,3))
    }else{
        lambda2_vec <- 0
    }
    
    if(lambda_z & !commonSupp){
        lambdaZ <- sort( unique( c(0, 1e-6, 1e-5, 1e-4, 1e-3,
                                   exp(-seq(0,5, length = 8)),
                                   1:3) ),
                         decreasing = TRUE ) 
        
        lambda2_vec <- 10^(seq(-5,3))
    }else{
        lambda2_vec <- 0
    }
    
    
    
    
    
}