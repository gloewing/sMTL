#' sparseCV_L0: cross-validation functions. For internal package use only.
#' @param data Matrix with outcome and design matrix
#' @param tune.grid A data.frame of tuning values
#' @param hoso String specifying tuning type
#' @param nfolds String or integer specifying number of folds
#' @param juliaFnPath String specifying path to Julia binary
#' @param trainingStudy Integer specifying index of training study
#' @param messageInd Boolean for message printing
#' @param LSitr Integer specifying do <LSitr> local search iterations on parameter values where we do actually do LS; NA does no local search
#' @param LSspc Integer specifying number of hyperparameters to conduct local search: conduct local search every <LSspc>^th iteration. NA does no local search
#' @param maxIter Integer specifying max iterations of coordinate descent
#' @import JuliaConnectoR
#' @export

# this function is called upon by sparseCV above and does L0 tuning with L2 penalty with various
# ways of generating train and validation sets
sparseL0Tn_iht <- function(data,
                     tune.grid,
                     hoso = "hoso", # could balancedCV # study balanced CV necessary if K =2
                     nfolds = "K",
                     juliaFnPath = "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/",
                     trainingStudy = NA, # this is the index of the training study for training each study of an ensemble and testing on all the rest
                     messageInd = FALSE, # if true then show messages about tuning status
                     LSitr = 50, # do <LSitr> local search iterations on parameter values where we do actually do LS
                     LSspc = 1, # do local search every <LSspc>^th iteration
                     maxIter = 2500
                     ){

  # L0 regression
  suppressWarnings( rm(L0_regression ) )
  
  # covariate indices
  Xindx <- which(!names(data) %in% c("Study", "Y")) 
  
  if( !exists("L0_regression") )  L0_regression <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # IHT

  # rename studies from 1:K
  num.trainStudy <- K <- length(unique(data$Study))
  data$Study <- as.numeric( as.factor( data$Study) ) # replace with new study labels from 1:K
  studyVec <- studies <- unique(data$Study)
  ########################

  ########################################
  # create folds depending on hoso type
  ########################################
  if(nfolds == "K")  nfolds <- K

  indxL <- HOOL <- vector(length = nfolds, "list") # lists of indices for training

  if(hoso == "hoso"){
    # each fold trains with all studies except hold out one study
    for(study in 1:num.trainStudy){

      indxL[[study]] <- which(data$Study != study) # indices of studies to train on
      HOOL[[study]] <- which(data$Study == study) # indices of study to hold out
    }

  }else if(hoso == "balancedCV"){
    # balance based on studies
    HOOL <- caret::createFolds(factor(data$Study), k = nfolds) # make folds with balanced studies
    allRows <- 1:nrow(data)

    for(study in 1:nfolds){
      Hind <- HOOL[[study]] # indices of study to hold out
      indxL[[study]] <- allRows[-Hind]
    }
  }else if(hoso == "out"){
    # out of study validation for training each individual study of an ensemble
    # train on selected study (trainingStudy) and validate on all the others merged together
    nfolds <- 1
    indxL <- HOOL <- vector(length = nfolds, "list") # lists of indices for training

    indxL[[1]] <- which(data$Study == trainingStudy) # indices of studies to train on
    HOOL[[1]] <- which(data$Study != trainingStudy) # indices of all other studies (used to test on)

  }
  ########################
  # start merged L0 tuning
  ########################
    # hold one study out on the merged dataset--used for Merged tuning

    lambdaMat <- matrix(nrow = nfolds, ncol = ncol(tune.grid) ) # stores best lambdas
    rmseMat <- matrix(nrow = nfolds, ncol = nrow(tune.grid) ) # store RMSEs for current study

    # random initialization of beta value
    b <- b_init <- rep(0, ncol(data) - 1) #rnorm( ncol(data) - 1 )

    for(study in 1:nfolds){

      if(messageInd)    message(paste0("Tuning fold: ", study, ", of ", nfolds, " folds"))

      b <- b_init # initialize with first fold of highest rho
      
      indxList <- indxL[[study]] # indices of studies to train on
      HOOList <- HOOL[[study]] # indices of study to hold out

      nk <- length(indxList) # sample size of kth study
      rhoVec <- unique(tune.grid$rho)

      ##########################
      # HOO CV
      ##########################
      totParams <- nrow(tune.grid) # total parameters
      # max eigenvalue for Lipschitz constant

      for(j in 1:length(rhoVec)){

        if(messageInd)    message(paste0("Tuning fold: ", study, ", sparsity (s) ", j, " of ", length(rhoVec)))

        # check to see if L0Learn has a solution here only if
        #rho is different (i.e., we dont already have a good solution at this cardinality)

        rhoIndx <- which(tune.grid$rho == rhoVec[j]) # indices of lambda values with this rho value
        lambdaVec <- tune.grid$lambda[rhoIndx]

        # make sequence of number of local iterations
        if( is.na(LSitr) | is.na(LSspc) ){
          # if no local search at all
          localIter <- rep(0, length(lambdaVec) )
        }else{
          # if local search
          localIter <- rep(0, length(lambdaVec) )
          LSindx <- seq(1, length(lambdaVec), by = LSspc) # make a sequence thats every LSspc^th
          localIter[LSindx] <- LSitr # 0 everywhere except every LSspc^th element = LSitr
          localIter[length(localIter)] <- LSitr # make last element also have local search so closer to every LSspc^th element
        }

        # warm start
        b = L0_regression(X = as.matrix(data[ indxList, Xindx  ]),
                     y = as.numeric(data$Y[  indxList  ]),
                     rho = as.integer( min( rhoVec[j], length(Xindx) - 1) ),
                     beta = as.matrix(b),
                     lambda = as.numeric( max(lambdaVec) ),
                     scale = TRUE,
                     maxIter = as.integer(1000),
                     localIter = as.integer(0)
                    )
        
        if(study == 1 & j == 1)       b_init <- b  # if first fold and first study initialize warm start to warm start
        
        # L0 constrained ridge estimator
        fit = L0_regression(X = as.matrix(data[ indxList, Xindx  ]),
                      y = as.numeric( data$Y[  indxList  ]) ,
                      rho = as.integer( rhoVec[j] ),
                      beta = as.matrix(b),
                      lambda = as.numeric(lambdaVec),
                      scale = TRUE,
                      maxIter = as.integer(maxIter),
                     localIter = as.integer(localIter)
                      )

        for(t in 1:length(lambdaVec) ){
          # iterate through fits
          tnIndx <- which(tune.grid$lambda == lambdaVec[t] & tune.grid$rho == rhoVec[j]) # indx of parameter values in tune.grid

          fitG <- fit[,t] # beta coefficient vector

          # predictions on merged dataset of all held out studies
          preds <- as.vector(  as.matrix( cbind(1, data[HOOList, Xindx]) ) %*% fitG   )
          rmseMat[study, tnIndx]  <- sqrt(mean( (preds - data$Y[HOOList] )^2  )) # rmse

        }



      }
    }

    rm(fit, preds)

    errorMean <- colMeans(rmseMat)
    testLambda <- tune.grid[ which.min(errorMean),]

    testLambda <- as.data.frame(testLambda)

    colnames(testLambda) <- colnames(tune.grid)
    rownames(rmseMat) <- paste0("fold_", 1:nrow(rmseMat) ) # put before line below: rbind( t(tune.grid), rmseMat)
    avg <- colMeans( rmseMat )

    rmseMat <- rbind( t(tune.grid), rmseMat)
    avg <- rbind( t(tune.grid), avg)
    colnames(rmseMat) <- colnames(avg) <- paste0("param", 1:nrow(tune.grid))

    if( length(rhoVec) > 1){
      se1 <- sMTL::seReturn(avg)
    }else{
      se1 <- NA
    }
    return( list(best = testLambda, best.1se = se1, rmse = rmseMat, avg = avg) )

}
