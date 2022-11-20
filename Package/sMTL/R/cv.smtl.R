#' cv.smtl: cross-validation function
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

cv.smtl = function(y, 
                X, 
                study = NA, 
                grid,
                nfolds = NA,
                commonSupp = FALSE,
                multiTask = TRUE, # only used if study indices provided, then use this to distinguish between a "hoso" and " multiTask" tuning 
                maxIter = 2500,
                LocSrch_skip = 1,
                LocSrch_maxIter = 10,
                messageInd = FALSE,
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
    
    convex_flag <- FALSE # whether to solve the convex problem for ALL solutions
    
    if(any(grid$s >= p)){
        message(paste("Some values of s (",max(grid$s),") (support size) are >= than number of covaraites (",p,"). s is set to p",sep=""))
        sp_index <- which(grid$s >= p)
        grid$s[sp_index] <- p # set to p
        if(  length(grid$lambda_z) > 0   )    grid$lambda_z[sp_index] <- 0 # cannot have s = p and lambda_z > 0
        
        if(all(grid$s >= p)){
            # if all values are convex problem, then set it up to trigger convex version for all
            commonSupp <- TRUE
            LocSrch_maxIter <- 0
            convex_flag <- TRUE # whether to solve the convex problem for ALL solutions
            independent.regs <- FALSE # cannot be independent
        }     
        
    }     
    
    # add regularization if there isn't any
    sp_index <- which(grid$lambda_1 == 0 & grid$lambda_2 == 0)
    if(length(sp_index) > 1)    grid$lambda_1[sp_index] <- 1e-7 # add small penalty to ensure unique solution
    
    
    if(anyNA(grid$s)){
        message(paste("Some values of s (",s,") (support size) are not specified (are NA). These rows have been removed",sep=""))
        
        grid <- grid[!is.na(grid$s), ] # remove rows with NAs
        
        if(nrow(grid) < 2)  stop(paste("Fewer than 2 rows in tuning grid after removing NAs",sep=""))
    }
    
    
    grid[is.na(grid)] <- 0 # remove NAs 
    if(nrow(grid) < 2)  stop(paste("Fewer than 2 values of rows in tuning grid",sep=""))
    
    # find path of sMTL package
    juliaFnPath <- paste0( .libPaths("sMTL"), "/sMTL/julia/" )
    
    # catch errors
    if(!is.na(nfolds)) nfolds <- round(nfolds)
    maxIter <- round(maxIter)
    
    if(any(grid$s >= 50) & LocSrch_maxIter > 0)   message(paste("Some values of s (",max(grid$s),") (support size) are >= than 50 and local search is being used. We recommend setting LocSrch_maxIter = 0, or if local search is needed, set LocSrch_skip >= 5",sep=""))
    
    if(is.na(LocSrch_maxIter))        LocSrch_maxIter <- as.integer( 0 )
    if(is.na(LocSrch_skip))           LocSrch_skip <- as.integer( 1 )
    
    if(is.numeric(LocSrch_maxIter))   LocSrch_maxIter <- as.integer( round(LocSrch_maxIter) )
    if(is.numeric(LocSrch_skip))      LocSrch_skip <- as.integer( round(LocSrch_skip) )
        
    if(!is.integer(LocSrch_maxIter) | LocSrch_maxIter < 0){
        message("LocSrch_maxIter must be an integer >= 0. We set it to default")
        LocSrch_maxIter <- 10
    }      
    if(!is.integer(LocSrch_skip) | LocSrch_skip < 1){
        message("LocSrch_skip must be an integer >= 1 We set it to default")
        LocSrch_skip <- 1
    }      
    
    
    # determine if problem is multiTask
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
            reg_type <- method <- "L0"
            #K <- 1
            K <- nfolds
        }else{
            # if study is given
            study <- as.integer( as.factor(study) ) # make integer for Julia type
            reg_type <- "multiStudy"
            K <- length( unique(study) )
            
            if(nfolds > K & !multiTask ){
                message("nfolds must be <= K, nfolds set to K ")
                nfolds <- K
            }else if(is.na(nfolds)){
                nfolds <- K
            }
        }
    }
    
    # if nfolds not specified and it is not multi-study
    if(is.na(nfolds) & reg_type != "multiStudy")   nfolds <- 5
    
    if (nfolds < 3 & reg_type != "multiStudy")   stop("nfolds must be bigger than 3; nfolds=10 recommended")
    
    # rename parameters for tuning function
    grid <- dplyr::rename(grid, rho = s)
    if( "lambda_1" %in% names(grid) ){
        grid <- dplyr::rename(grid, lambda1 = lambda_1)
    }     
    if( "lambda_2" %in% names(grid) ){
        grid <- dplyr::rename(grid, lambda2 = lambda_2) 
    }    
    
    ################################
    # find algorithm type ("method")
    ################################
    
    if( reg_type != "L0" ){
        if(commonSupp){
            
            if(messageInd)  message( paste("CV: Common Support", reg_type, "Dataset") )
            
            method <- ifelse( !any(grid$lambda2 > 0 ) & !convex_flag, 
                              "MS", # none have betaBar penalty -- MS2 and non-convex version
                              "MS2" # at least one has betaBar penalty
            )
            
        }else{
            
            if(messageInd)  message( paste("CV: Heterogeneous Support", reg_type, "Dataset") )
            
            # no common support
            method <- ifelse( independent.regs & ( all(grid$lambda2 == 0) & all(grid$lambda_z == 0) ), 
                              "MS_z3", # separate active sets
                              "MS_z"  # shared active sets
            )
            
        }
    }
    
    
    if( reg_type == "L0" | method == "MS" ){

        grid <- dplyr::rename(grid, lambda = lambda1)  # rename column to be consistent with tuning

        # remove names from tuning grid
        drops <- c("lambda2", "lambda_z")
        grid <- unique( grid[, !(names(grid) %in% drops) ] )
    }else if(method == "MS2"){
        
        # remove names from tuning grid
        drops <- c("lambda_z")
        grid <- unique( grid[, !(names(grid) %in% drops) ] )
        
    }
    
    # if there is no cardinality constraint
    if( all(grid$rho >= p) ){
        method <- "MS2" # no active sets so it should be much faster
        commonSupp <- TRUE
        
        # remove names from tuning grid
        drops <- c("lambda_z")
        grid <- unique( grid[, !(names(grid) %in% drops) ] )
    }
    
    
    if(reg_type == "multiLabel"){
        
        tuneStyle <- "multiTask" # tuning style not regression type
        
        # concatenate into one dataframe
        X <- cbind(y, X)
        rm(y)
        
        # rename
        colnames(X)[1:K] <- paste0("Y_", 1:K)
        colnames(X)[-seq(1,K)] <- paste0("x_", 1:(ncol(X) - K) )
        X <- data.frame(X)
        
        tuneMS <- sMTL::sparseCV_MT(data = X,
                                 tune.grid = grid,
                                 hoso = tuneStyle, # could balancedCV (study balanced CV necessary if K =2)
                                 method = method, # could be L0 for sparse regression or MS # for multi study
                                 nfolds = nfolds,
                                 juliaFnPath = juliaFnPath,
                                 messageInd = messageInd,
                                 LSitr = LocSrch_maxIter, 
                                 LSspc = LocSrch_skip,
                                 maxIter = maxIter
                                )
        
        return(tuneMS)
        
    }else if(reg_type == "multiStudy"){
        
        tuneStyle <- ifelse(multiTask, "multiTask", "hoso") # indicator to determine TUNING style not regression style
        
        # concatenate into one dataframe
        X <- cbind(study, y, X)
        rm(y)
        
        # rename
        colnames(X)[1:2] <- c("Study", "Y")
        colnames(X)[-seq(1,2)] <- paste0("x_", 1:(ncol(X) - 2) )
        X <- data.frame(X)
        
        tuneMS <- sMTL::sparseCV(data = X,
                                 tune.grid = grid,
                                 hoso = tuneStyle, # could balancedCV (study balanced CV necessary if K =2)
                                 method = method, # could be L0 for sparse regression or MS # for multi study
                                 nfolds = nfolds,
                                 juliaFnPath = juliaFnPath,
                                 messageInd = messageInd,
                                 LSitr = LocSrch_maxIter, 
                                 LSspc = LocSrch_skip,
                                 maxIter = maxIter
                                )
        
        return(tuneMS)
        
    }else if(reg_type == "L0"){
        
        tuneStyle <- ifelse(K > 2, "hoso", "balancedCV") # can only do hoso CV if K > 2
        
        # if merged indicator given, then randomly draw studies
        if( is.na(study) ){
            study <- vector(length = nrow(X) )
            indx <- caret::createFolds(1:nrow(X), k = nfolds) # randomly assign folds ("study indices")
            if(nfolds > 2)    tuneStyle <- "hoso" # use hoso here
                
            for(kk in 1:nfolds){
                study[ indx[[kk]]  ] <- kk
            }
            
            if(messageInd)  message( paste("CV: Merged L0 Regression") )
            
        }else{
            if(messageInd)  message( paste("CV: Merged L0 Regression tuned with Hold-One-Study-Out CV") )
            
        }
        
        
        # concatenate into one dataframe
        X <- cbind(study, y, X)
        rm(y)
        
        # rename
        colnames(X)[1:2] <- c("Study", "Y")
        colnames(X)[-seq(1,2)] <- paste0("x_", 1:(ncol(X) - 2) )
        X <- data.frame(X)
        
        tuneMS <- sMTL::sparseCV(data = X,
                                 tune.grid = grid,
                                 hoso = tuneStyle, # could balancedCV (study balanced CV necessary if K =2)
                                 method = "L0", # could be L0 for sparse regression or MS # for multi study
                                 nfolds = nfolds,
                                 juliaFnPath = juliaFnPath,
                                 messageInd = TRUE,
                                 LSitr = LocSrch_maxIter, 
                                 LSspc = LocSrch_skip,
                                 maxIter = maxIter
                                )
        
        return(tuneMS)
        
    }
    
    return(tuneMS)
    
}
    
    