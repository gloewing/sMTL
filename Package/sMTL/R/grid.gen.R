#' grid.gen: generate grid for cross-validation function. For internal package use only.
#'
#' @param y A numeric vector or matrix of outcomes
#' @param p An integer of covariate dimension
#' @param study An integer vector of task IDs
#' @param lambda_1 A boolean
#' @param lambda_2 A boolean
#' @param lambda_z A boolean
#' @param commonSupp A boolean
#' @param multiTask A boolean
#' @return A dataframe
#' @import JuliaConnectoR
#' @import dplyr
#' @export

grid.gen = function(y,
                    p,
                   study = NA, 
                   lambda_1 = TRUE,
                   lambda_2 = FALSE,
                   lambda_z = TRUE,
                   commonSupp = FALSE,
                   multiTask = TRUE # only used if study indices provided, then use this to distinguish between a "hoso" and " multiTask" tuning 
) {
    
    ##################
    # sparsity level
    ##################
    # sparsity parameter depends on dimension of X
    if(p <= 50){
        s <- c(2, 4, 6, 10, 15, 20)
    }else if(p > 50 & p <= 500){
        s <- c(5, 10, 15, 20, 30)
    }else if(p > 500){
        s <- c(5, 10, 15, 25, 40 )
    }

    ##################
    # problem type
    ##################
    
    # if common support set multi-study paramters to FALSE
    if(commonSupp)     lambda_z <- lambda_2 <- FALSE
    
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
            lambda_2 <- lambda_z <- FALSE # these penalties are meaningless for single task problems
        }else{
            # if study is given
            study <- as.integer( as.factor(study) ) # make integer for Julia type
            reg_type <- "multiStudy"
            K <- length( unique(study) )
            
        }
    }
    
    ##################
    # create grid
    ##################
    # ridge penalty
    if(lambda_1)        lambda_1 <- c(1e-7, 1e-5, 1e-3, 1e-1, 1) 
 
    # bbar penalty
    if(lambda_2)        lambda_2 <- c(0,
                                      10^(seq(-3,-1)),
                                      exp( seq(2,6, length = 6)) )
    # zBar penalty
    if(lambda_z)        lambda_z <- sort( unique( c(0, 1e-6, 1e-5, 1e-4, 1e-3,
                                                    exp(-seq(0,5, length = 8)),
                                                    1, 3) ),    decreasing = TRUE )
    
    if(!lambda_1)        lambda_1 <- 0
    if(!lambda_2)        lambda_2 <- 0
    if(!lambda_z)        lambda_z <- 0
    
    grid <- as.data.frame(  expand.grid( lambda_1, lambda_2, lambda_z, s) )
    colnames(grid) <- c("lambda_1", "lambda_2", "lambda_z", "s")
    
    # scale z
    lambdaZScale <- sMTL::rhoScale(K = K, 
                                     p = p, 
                                     rhoVec = grid$s, 
                                     itrs = 5000,
                                     seed = 1)
    
    grid <- sMTL::tuneZscale(tune.grid = grid, 
                             rhoScale = lambdaZScale)
    
    # order correctly
    grid <- grid[  order(-grid$s,
                         -grid$lambda_2,
                         grid$lambda_1,
                         -grid$lambda_z,
                         decreasing=TRUE),     ]

    
    return( expand.grid(grid) )
    
}