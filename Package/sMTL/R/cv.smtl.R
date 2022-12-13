#' cv.smtl: cross-validation function
#'
#' @param y A numeric outcome vector or matrix (for multi-label problems)
#' @param X A design (feature) matrix
#' @param study An integer vector specifying the task ID
#' @param grid A dataframe with column names "s", "lambda_1", "lambda_2" and "lambda_z" (if commonSupp = FALSE) with tuning values
#' @param nfolds An integer specifying number of CV folds
#' @param commonSupp A boolean specifying whether the task models should have the same support
#' @param multiTask A boolean only used if study/task indices are provided: used to distinguish between a Multi-Task Learning Tuning (TRUE) or Domain Generalization Tuning (FALSE)
#' @param lambda_1 An optional boolean: if a grid is not provided, then set to TRUE if you want an automatic grid to be generated with non-zero values for this hyperparameter
#' @param lambda_2 An optional boolean: if a grid is not provided, then set to TRUE if you want an automatic grid to be generated with non-zero values for this hyperparameter
#' @param lambda_z An optional boolean: if a grid is not provided, then set to TRUE if you want an automatic grid to be generated with non-zero values for this hyperparameter
#' @param maxIter An integer specifying the maximum number of coordinate descent iterations
#' @param LocSrch_skip An integer specifying whether to use local search at every tuning value (set to 1), every other value (set to 2), every third (set to 3),...
#' @param LocSrch_maxIter An integer specifying the maximum number of local search iterations
#' @param messageInd A boolean (verbose) of whether to print messages
#' @param independent.regs A boolean of whether models are completely indpendent (only set to TRUE for benchmarks)
#' @return A list
#' @examples
#' 
#' #####################################################################################
#' ##### simulate data
#' #####################################################################################
#' set.seed(1) # fix the seed to get a reproducible result
#' K <- 4 # number of datasets 
#' p <- 100 # covariate dimension
#' s <- 5 # support size
#' q <- 7 # size of subset of covariates that can be non-zero for any task
#' n_k <- 50 # task sample size
#' N <- n_k * p # full dataset samplesize
#' X <- matrix( rnorm(N * p), nrow = N, ncol=p) # full design matrix
#' B <- matrix(1 + rnorm(K * (p+1) ), nrow = p + 1, ncol = K) # betas before making sparse
#' Z <- matrix(0, nrow = p, ncol = K) # matrix of supports
#' y <- vector(length = N) # outcome vector
#' 
#' # randomly sample support to make betas sparse
#' for(j in 1:K)     Z[1:q, j] <- sample( c( rep(1,s), rep(0, q - s) ), q, replace = FALSE )
#' B[-1,] <- B[-1,] * Z # make betas sparse and ensure all models have an intercept
#' 
#' task <- rep(1:K, each = n_k) # vector of task labels (indices)
#' 
#' # iterate through and make each task specific dataset
#' for(j in 1:K){
#'     indx <- which(task == j) # indices of task
#'     e <- rnorm(n_k)
#'     y[indx] <- B[1, j] + X[indx,] %*% B[-1,j] + e
#'     }
#'     
#' colnames(B) <- paste0("beta_", 1:K)
#' rownames(B) <- paste0("X_", 1:(p+1))
#'     
#' print("Betas")
#' print(round(B[1:8,],2))
#'     
#'     ###########################
#'     # custom tuning grid
#'     ###########################
#'     grid <- data.frame(s = c(4, 4, 5, 5), 
#'                   lambda_1 = c(0.01, 0.1, 0.01, 0.1), 
#'                   lambda_2 = rep(0, 4), 
#'                   lambda_z = c(0.01, 0.1, 0.01, 0.1))
#'     
#'     #################################################
#'     # cross validation with custom tuning grid
#'     ##################################################
#'     \dontrun{
#'     tn <- cv.smtl(y = y, 
#'                   X = X, 
#'                   study = task, 
#'                   commonSupp = FALSE,
#'                   grid = grid,
#'                   nfolds = 5,
#'                   multiTask = FALSE) 
#'                   
#'      # model fitting
#'      mod <- sMTL::smtl(y = y, 
#'                    X = X, 
#'                    study = task, 
#'                    s = tn$best.1se$s, 
#'                    commonSupp = TRUE,
#'                    lambda_1 = tn$best.1se$lambda_1,
#'                    lambda_z = tn$best.1se$lambda_z)
#'     
#'     ######################################################
#'     # cross validation with automatically generated grid
#'     #######################################################
#'     tn <- cv.smtl(y = y, 
#'                   X = X, 
#'                   study = task, 
#'                   commonSupp = FALSE,
#'                   lambda_1 = TRUE,
#'                   lambda_w = FALSE,
#'                   lambda_z = TRUE,
#'                   nfolds = 5,
#'                   multiTask = FALSE) 
#'     
#'      # model fitting
#'      mod <- sMTL::smtl(y = y, 
#'                    X = X, 
#'                    study = task, 
#'                    s = tn$best.1se$s, 
#'                    commonSupp = TRUE,
#'                    lambda_1 = tn$best.1se$lambda_1,
#'                    lambda_z = tn$best.1se$lambda_z)
#'                    
#'      print(round(mod$beta[1:8,],2))
#'                    }
#'                    
#'     
#' @import JuliaConnectoR
#' @export

cv.smtl = function(y, 
                X, 
                study = NA, 
                grid = NA,
                nfolds = NA,
                commonSupp = FALSE,
                multiTask = TRUE, # only used if study indices provided, then use this to distinguish between a "hoso" and " multiTask" tuning 
                lambda_1 = TRUE, # a flag that is used if grid = NA to generate a tuning grid with ridge penalty
                lambda_2 = FALSE, # a flag that is used if grid = NA to generate a tuning grid with Bbar penalty
                lambda_z = TRUE, # a flag that is used if grid = NA to generate a tuning grid with Zbar penalty
                maxIter = 2500,
                LocSrch_skip = 1,
                LocSrch_maxIter = 10,
                messageInd = FALSE,
                independent.regs = FALSE # shared active sets
) {
    
    np <- dim(X)
    nobs <- as.integer(np[1])
    p <- as.integer(np[2])
    
    lambda1 <- s <- NULL # global variable declaration for CRAN checks
    
    #########################################
    # tuning grid - generate if not provided
    #########################################
    if(is.na(grid)[1])      grid <- grid.gen(y = y,
                                             p = p,
                                             study = NA, 
                                             lambda_1 = lambda_1,
                                             lambda_2 = lambda_2,
                                             lambda_z = lambda_z,
                                             commonSupp = commonSupp,
                                             multiTask = multiTask) # only used if study indices provided, then use this to distinguish between a "hoso" and " multiTask" tuning 

    
    ###################
    # sanity checks
    ###################
    
    ##check dims
    if(is.null(np)|(np[2]<=1))   stop("X should be a matrix with 2 or more columns")
    
    dimy=dim(y)
    nrowy=ifelse(is.null(dimy),length(y),dimy[1])
    if(nrowy!=nobs)  stop(paste("number of observations in y (",nrowy,") not equal to the number of rows of x (",nobs,")",sep=""))
    
    convex_flag <- FALSE # whether to solve the convex problem for ALL solutions
    
    if(any(grid$s >= p)){
        message(paste("Some values of s (",max(grid$s),") (support size) are >= than number of covariates (",p,"). s is set to p",sep=""))
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
    
    ####################################
    # determine if problem is multiTask
    ####################################
    if( is.matrix(y) ){
        
        if( any( apply(y, 2, stats::var)  == 0 ) )   stop("At least one task's y is constant")
        
        
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
        
        if( stats::var(y) == 0 )   stop("y is constant")
        
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
                message("Warning: nfolds must be <= K, nfolds set to K ")
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
    
    