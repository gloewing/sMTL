#' smtl: make model-fitting function
#'
#' @param y A numeric vector
#' @param X A matrix
#' @param s An integer
#' @param commonSupp A boolean
#' @param lambda_1 A numeric vector 
#' @param lambda_2 A numeric vector 
#' @param lambda_z A numeric vector 
#' @param scale A boolean
#' @param maxIter An integer
#' @param LocSrch_maxIter An integer
#' @param independent.regs A boolean
#' @param model A boolean
#' @param messageInd A boolean
#' @return A model object (list)
#' @examples
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Installed and Julia Path is Known ######
#' #####################################################################################
#' smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", installJulia = FALSE, installPackages = FALSE)"
#' 
#' #######################################################################################################
#' ##### If you have run smtl_setup() before, then path specification shouldn't be necessary ######
#' #######################################################################################################
#' smtl_setup(path = NULL, installJulia = FALSE, installPackages = FALSE)"
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Not Installed   ######
#' #####################################################################################
#' smtl_setup(path = NULL, installJulia = TRUE, installPackages = FALSE)"
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Installed But Packages NEED INSTALLATION  ######
#' #####################################################################################
#' smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", installJulia = TRUE, installPackages = TRUE)"
#' @import JuliaConnectoR
#' @export

smtl = function(y, 
                X, 
                study = NA, 
                s, 
                commonSupp = FALSE,
                warmStart = TRUE,
                lambda_1 = 0, 
                lambda_2 = 0, 
                lambda_z = 0, 
                scale = TRUE,
                maxIter = 10000,
                LocSrch_maxIter = 50,
                messageInd = TRUE,
                model = TRUE,
                independent.regs = FALSE # shared active sets
                ) {
    
    ###################
    # sanity checks
    ###################
    np <- dim(X)
    nobs <- as.integer(np[1])
    p <- as.integer(np[2])
    
    ##check dims
    if(is.null(np)|(np[2]<=1))   stop("X should be a matrix with 2 or more columns")
    
    dimy=dim(y)
    nrowy=ifelse(is.null(dimy),length(y),dimy[1])
    if(nrowy!=nobs)  stop(paste("number of observations in y (",nrowy,") not equal to the number of rows of x (",nobs,")",sep=""))
    
    convex_flag <- FALSE # indicator of whethere s>=p
        
    if(any(s >= p)){
        
        message(paste("s (",s,") (support size), is >= number of covaraites (",p,"). s is set to p. commonSupp set to TRUE. Solving non-sparse convex problem.",sep=""))
        s <- p # set to p
        commonSupp <- TRUE
        lambda_z <- 0 # cannot have convex version and lambda_z > 0
        convex_flag <- TRUE  # indicator of whethere s>=p
        LocSrch_maxIter <- 0 # no local search 
        independent.regs <- FALSE # cannot be independent
        }   
    
    
    # find path of sMTL package
    juliaFnPath <- paste0( .libPaths("sMTL"), "/sMTL/julia/" )
    
    # catch errors
    maxIter <- round(maxIter)
    
    if(is.numeric(LocSrch_maxIter))   LocSrch_maxIter <- as.integer( round(LocSrch_maxIter) )
    if(is.na(LocSrch_maxIter))        LocSrch_maxIter <- as.integer( 0 )
    
    if(!is.integer(LocSrch_maxIter) | LocSrch_maxIter < 0){
        message("LocSrch_maxIter must be an integer >= 0. We set it to default")
        LocSrch_maxIter <- 50
    }      
    
    # find path of sMTL package
    juliaFnPath <- paste0( .libPaths("sMTL"), "/sMTL/julia/" )
    
    AS_multiplier <- 3 # warm start multiplier for number of non-zeros
    
    # determine if problem is multiLabel
    if( is.matrix(y) ){
        
        if( any( apply(y, 2, var)  == 0 ) )   stop("At least one task's y is constant")
        
        # make sure y is a matrix with multiple columns
        if( ncol(y) > 1 ){
            reg_type <- "multiLabel"
            K <- ncol(y)
        }else{
            # if only 1 column, then coerce into vector
            y <- as.numeric(y)
        }    

    }
    
    if( is.vector(y) ){
        
        if( var(y) == 0 )   stop("y is constant")
        
        # y is a vector and no studies given then just L0 problem
        if( anyNA(study) ){
            reg_type <- "L0"
            K <- 1
        }else{
            # if study is given
            reg_type <- "multiStudy"
            study <- as.integer( as.factor( study ) )
            K <- length( unique(study) )
        }
    }
    
    #####################
    # order tuning values
    #####################
    # ensure grids are of same length
    gridLen <- max( c(length(lambda_1),
                      length(lambda_2),
                      length(lambda_z)
                      )
                    )    
    
    # remove any NAs
    lambda_1[(is.na(lambda_1))] <- 0
    lambda_2[(is.na(lambda_2))] <- 0
    lambda_z[(is.na(lambda_z))] <- 0
    
    # if there is only value, repeat it for the entire grid
    if( length(lambda_1) == 1)   lambda_1 <- rep(lambda_1, gridLen)
    if( length(lambda_2) == 1)   lambda_2 <- rep(lambda_2, gridLen)
    if( length(lambda_z) == 1)   lambda_z <- rep(lambda_z, gridLen)
    
    # check if they are of the same length
    if( var( c(length(lambda_1),
               length(lambda_2),
               length(lambda_z)) 
             ) != 0 )    stop("Lengths of vectors lambda_1, lambda_2 and lambda_z must be of the same length")
    
    grid <- data.frame(lambda_1 = lambda_1,
                       lambda_2 = lambda_2,
                       lambda_z = lambda_z)
    
    grid <- grid[  order(grid$lambda_1,
                         -grid$lambda_2,
                         -grid$lambda_z,
                         decreasing=TRUE),  ]
    
    # add regularization if there isn't any
    sp_index <- which(grid$lambda_1 == 0 & grid$lambda_2 == 0)
    if(length(sp_index) > 1)    grid$lambda_1[sp_index] <- 1e-7 # add small penalty to ensure unique solution
    
    
    lambda_1 <- grid$lambda_1
    lambda_2 <- grid$lambda_2
    lambda_z <- grid$lambda_z
    
    # if local iterations not specified for all tuning values, use the first one for all values
    LocSrch_maxIter <- rep(LocSrch_maxIter[1], nrow(grid) ) 
    
    #####################
    # return object
    #####################
    fit_list <- list(   beta = NA,
                        reg_type = reg_type,
                        K = K,
                        s = s, 
                        commonSupp = commonSupp,
                        warmStart = warmStart,
                        grid = grid,
                        scale = scale,
                        maxIter = maxIter,
                        LocSrch_maxIter = LocSrch_maxIter,
                        independent.regs = independent.regs, 
                        AS_multiplier = AS_multiplier
                    )
    
    if(model){
        fit_list$X_train <- X
        fit_list$y_train <- y
    }   
    
    rm(grid)
    #####################
    # L0 regression
    #####################
    if( reg_type == "L0" ){
        
        # L0 regression
        
        if( !exists("L0_reg") )  L0_reg <- JuliaConnectoR::juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # sparseReg
        
        suppressWarnings(  if( anyNA(warmStart) | warmStart == TRUE ){
            b <- rep(0, ncol(X) + 1) 
        }else if( is.vector(warmStart) ){
            b <- warmStart
        }
        
        )
        
        if(warmStart & s < p){
            
            if(messageInd)   message("Warm Start Model Running")
            
            b <- L0_reg(X = as.matrix( X ),
                          y = as.numeric( y ),
                          rho = as.integer( min(AS_multiplier * s, p) ),
                          beta = as.matrix(b),
                          lambda = as.numeric(max(lambda_1)),
                          scale = scale,
                          maxIter = as.integer( 500 ),
                          localIter = as.integer(0)
            )
            
        }
        
        if(messageInd)   message("Single L0 Regression")
        
        fit_list$beta <- L0_reg(X = as.matrix( X ),
                      y = as.numeric( y ),
                      rho = as.integer(s),
                      beta = as.matrix(b),
                      lambda = as.numeric(lambda_1),
                      scale = scale,
                      maxIter = as.integer( maxIter ),
                      localIter = as.integer(LocSrch_maxIter)
                        )
        
        dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
        dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
        
        return(fit_list)
        
        
    #####################
    # multiStudy
    #####################
    }else if( reg_type == "multiStudy" ){
        
        # warm start with matrix of 0s
        suppressWarnings( if( anyNA(warmStart) | warmStart == TRUE ){
            b <- matrix(0, nrow = ncol(X) + 1, ncol = K) 
        }else if(is.matrix(warmStart)){
            b <- warmStart
        }
        
        )
        
        #*********************************
        # multiStudy: Common Support
        #*********************************
        if(commonSupp){
            
            if( !any(lambda_2 > 0) & s < p){
                
                # none have betaBar penalty
                if( !exists("L0_MS") )  L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune.jl") )
                
                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS(X = as.matrix( X ),
                                      y = as.numeric( y ),
                                      rho = as.integer( min(AS_multiplier * s, p) ),
                                      study = as.integer(study), # these are the study labels ordered appropriately for this fold
                                      beta = as.matrix(b),
                                      lambda = as.numeric(max(lambda_1)),
                                      scale = scale,
                                      maxIter = as.integer( 500 ),
                                      localIter = as.integer(0)
                                )
                    
                }
                
                if(messageInd)   message("Common Support Multi-Study")
                
                fit_list$beta <- L0_MS(X = as.matrix( X ),
                                y = as.numeric( y ),
                                rho = as.integer(s),
                                study = as.integer(study), # these are the study labels ordered appropriately for this fold
                                beta = as.matrix(b),
                                lambda = as.numeric(lambda_1),
                                scale = scale,
                                maxIter = as.integer( maxIter ),
                                localIter = as.integer(LocSrch_maxIter)
                )
                
                dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
                dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
                
                return(fit_list)
                
            }else{
                
                # if some have betaBar penalty or s == p
                if( !exists("L0_MS2") )  L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune.jl") ) 
                
                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS2(X = as.matrix( X ),
                                       y = as.numeric( y ),
                                       rho = as.integer( min(AS_multiplier * s, p) ),
                                       study = as.integer(study), # these are the study labels ordered appropriately for this fold
                                       beta = as.matrix(b),
                                       lambda1 = as.numeric(max(lambda_1)),
                                        lambda2 = as.numeric(0),
                                       scale = scale,
                                       maxIter = as.integer( 500 ),
                                       localIter = as.integer(0)
                    )

                }
                
                if(messageInd)   message("Common Support Multi-Study with Beta-Bar Penalty")
                
                fit_list$beta <- L0_MS2(X = as.matrix( X ),
                             y = as.numeric( y ),
                             rho = as.integer(s),
                             study = as.integer(study), # these are the study labels ordered appropriately for this fold
                             beta = as.matrix(b),
                             lambda1 = as.numeric(lambda_1),
                             lambda2 = as.numeric(lambda_2),
                             scale = scale,
                             maxIter = as.integer( maxIter ),
                             localIter = as.integer(LocSrch_maxIter)
                            )
                
            }
            
            dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
            dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
            
            return(fit_list)
            
            #**********************************
            # multiStudy: Heterogeneous Support
            #**********************************
        }else{
            

            if( independent.regs & ( all(lambda_2 == 0) & all(lambda_z == 0) ) ){
                
                if(messageInd)   message("Heterogeneous Support Multi-Study with Separate Active Sets")
                
                # # no shared AS (only use for completely separate LO)
                if( !exists("L0_MS_z3") )   L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study
                # # 

                if(warmStart & s < p){
                    
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS_z3(X = as.matrix( X ),
                                   y = as.numeric( y ),
                                   rho = as.integer( min(AS_multiplier * s, p) ),
                                   study = study, # these are the study labels ordered appropriately for this fold
                                   beta = as.matrix(b),
                                   lambda1 = max( as.numeric(lambda_1) ),
                                   lambda2 = 0,
                                   lambda_z = 0,
                                   scale = scale,
                                   maxIter = as.integer( 500 ),
                                   localIter = as.integer(0)
                    )
                }
                
                fit_list$beta <- L0_MS_z3(X = as.matrix( X ),
                                  y = as.numeric( y ),
                                  rho = as.integer(s),
                                  study = as.integer(study), # these are the study labels ordered appropriately for this fold
                                  beta = as.matrix(b),
                                  lambda1 = as.numeric(lambda_1),
                                  lambda2 = as.numeric(lambda_2),
                                  lambda_z = as.numeric(lambda_z),
                                  scale = scale,
                                  maxIter = as.integer( maxIter ),
                                  localIter = as.integer(LocSrch_maxIter)
                )
                
            }else{
                
                # shared active sets OR sharing penalties
                if( !exists("L0_MS_z") )   L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                
                if(warmStart & s < p){
                    
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS_z(X = as.matrix( X ),
                                   y = as.numeric( y ),
                                   rho = as.integer( min(AS_multiplier * s, p) ),
                                   study = study, # these are the study labels ordered appropriately for this fold
                                   beta = as.matrix(b),
                                   lambda1 = max( as.numeric(lambda_1) ),
                                   lambda2 = 0,
                                   lambda_z = 0,
                                   scale = scale,
                                   maxIter = as.integer( 500 ),
                                   localIter = as.integer(0)
                                   )
                }
                
                if(messageInd)   message("Heterogeneous Support Multi-Study")
                
                fit_list$beta <- L0_MS_z(X = as.matrix( X ),
                               y = as.numeric( y ),
                               rho = as.integer(s),
                               study = as.integer(study), # these are the study labels ordered appropriately for this fold
                               beta = as.matrix(b),
                               lambda1 = as.numeric(lambda_1),
                               lambda2 = as.numeric(lambda_2),
                               lambda_z = as.numeric(lambda_z),
                               scale = scale,
                               maxIter = as.integer( maxIter ),
                               localIter = as.integer(LocSrch_maxIter)
                )
                
            }
            #L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
            
            dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
            dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
            
            return(fit_list)
        }
        #####################
        # multiLabel
        #####################
    }else if( reg_type == "multiLabel" ){
        

        # warm start with matrix of 0s
        suppressWarnings( if( anyNA(warmStart) | warmStart == TRUE ){
            b <- matrix(0, nrow = ncol(X) + 1, ncol = K) 
        }else if(is.matrix(warmStart)){
            b <- warmStart
        }
        
        )
        
        #*********************************
        # multiLabel: Common Support
        #*********************************
        if(commonSupp){
            
            if( !any(lambda_2 > 0) & s < p ){
                
                # none have betaBar penalty
                
                if( !exists("L0_MS_MT") )   L0_MS_MT <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
                
                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS_MT(X = as.matrix( X ),
                                    y = as.matrix( y ),
                                    rho = as.integer( min(AS_multiplier * s, p) ),
                                    study = NA, # these are the study labels ordered appropriately for this fold
                                    beta = as.matrix( b ) ,
                                    lambda = as.numeric(max(lambda_1)),
                                    scale = scale,
                                    maxIter = as.integer( 500 ),
                                    localIter = as.integer(0)
                    )
                    
                }
                
                if(messageInd)   message("Common Support Multi-Label")
                
                fit_list$beta <- L0_MS_MT(X = as.matrix( X ),
                             y = as.matrix( y ),
                             rho = as.integer(s),
                             study = NA, # these are the study labels ordered appropriately for this fold
                             beta = as.matrix( b ) ,
                             lambda = as.numeric(lambda_1),
                             scale = scale,
                             maxIter = as.integer( maxIter ),
                             localIter = as.integer(LocSrch_maxIter)
                )
                
                dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
                dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
                
                return(fit_list)
                
            }else{
                # if some have betaBar penalty or s == p
                
                if( !exists("L0_MS2_MT") )   L0_MS2_MT <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
                
                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS2_MT(X = as.matrix( X ),
                                          y = as.matrix( y ),
                                          rho = as.integer( min(AS_multiplier * s, p) ),
                                          study = NA, # these are the study labels ordered appropriately for this fold
                                          beta = as.matrix( b ) ,
                                          lambda1 = as.numeric(max(lambda_1)),
                                          lambda2 = 0,
                                          scale = scale,
                                          maxIter = as.integer( 500 ),
                                          localIter = as.integer(0)
                    )
                    
                }
                
                if(messageInd)   message("Common Support Multi-Label with Beta-Bar Penalty")
                
                fit_list$beta <- L0_MS2_MT(X = as.matrix( X ),
                              y = as.matrix( y ),
                              rho = as.integer(s),
                              study = NA, # these are the study labels ordered appropriately for this fold
                              beta = as.matrix(b),
                              lambda1 = as.numeric(lambda_1),
                              lambda2 = as.numeric(lambda_2),
                              scale = scale,
                              maxIter = as.integer( maxIter ),
                              localIter = as.integer(LocSrch_maxIter)
                )
                
            }
            
            dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
            dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
            
            return(fit_list)
            
            #**********************************
            # multiLabel: Heterogeneous Support
            #**********************************
        }else{
            
            if( independent.regs & ( all(lambda_2 == 0) & all(lambda_z == 0) ) ){
                
                # no shared AS (only use for completely separate LO)
                if( !exists("L0_MS_z3_MT") )   L0_MS_z3_MT <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study

                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS_z3_MT(X = as.matrix( X ),
                                   y = as.matrix( y ),
                                   rho = as.integer( min(AS_multiplier * s, p) ),
                                   study = NA, # these are the study labels ordered appropriately for this fold
                                   beta = as.matrix(b),
                                   lambda1 = as.numeric( max(lambda_1) ),
                                   lambda2 = as.numeric(0),
                                   lambda_z = as.numeric(0),
                                   scale = scale,
                                   maxIter = as.integer( 500 ),
                                   localIter = as.integer(0)
                    )
                }
                
                if(messageInd)   message("Heterogeneous Support Multi-Label with Separate Active Sets")
                
                
                fit_list$beta <- L0_MS_z3_MT(X = as.matrix( X ),
                                y = as.matrix( y ),
                                rho = as.integer(s),
                                study = NA, # these are the study labels ordered appropriately for this fold
                                beta = as.matrix(b),
                                lambda1 = as.numeric(lambda_1),
                                lambda2 = as.numeric(lambda_2),
                                lambda_z = as.numeric(lambda_z),
                                scale = scale,
                                maxIter = as.integer( maxIter ),
                                localIter = as.integer(LocSrch_maxIter)
                )
                
            }else{
                
                # shared active sets OR sharing penalties
                if( !exists("L0_MS_z_MT") )   L0_MS_z_MT <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                
                if(warmStart & s < p){
                    if(messageInd)   message("Warm Start Model Running")
                    
                    b <- L0_MS_z_MT(X = as.matrix( X ),
                                   y = as.matrix( y ),
                                   rho = as.integer( min(AS_multiplier * s, p) ),
                                   study = NA, # these are the study labels ordered appropriately for this fold
                                   beta = as.matrix(b),
                                   lambda1 = max( as.numeric(lambda_1) ),
                                   lambda2 = 0,
                                   lambda_z = 0,
                                   scale = scale,
                                   maxIter = as.integer( 500 ),
                                   localIter = as.integer(0)
                    )
                }
                
                if(messageInd)   message("Heterogeneous Support Multi-Label")
                
                fit_list$beta <- L0_MS_z_MT(X = as.matrix( X ),
                               y = as.matrix( y ),
                               rho = as.integer(s),
                               study = NA, # these are the study labels ordered appropriately for this fold
                               beta = as.matrix(b),
                               lambda1 = as.numeric(lambda_1),
                               lambda2 = as.numeric(lambda_2),
                               lambda_z = as.numeric(lambda_z),
                               scale = scale,
                               maxIter = as.integer( maxIter ),
                               localIter = as.integer(LocSrch_maxIter)
                )
                
            }
            #L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
            
            dimnames(fit_list$beta)[[2]] <- paste0("beta_", 1:(dim(fit_list$beta)[[2]]) ) # rename columns
            dimnames(fit_list$beta)[[1]] <- c("Intercept", paste0("V", 1:(dim(fit_list$beta)[[1]]) ) ) # rename rows
            
            return(fit_list)
        }
    }
    
}
    