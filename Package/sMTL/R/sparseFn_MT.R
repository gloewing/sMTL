#' sparseCV_MT: internal cross-validation functions. For internal package use only.
#' @param data Matrix with outcome and design matrix
#' @param tune.grid A data.frame of tuning values
#' @param hoso String specifying tuning type
#' @param method Sting specifying regression method
#' @param nfolds String or integer specifying number of folds
#' @param juliaFnPath String specifying path to Julia binary
#' @param messageInd Boolean for message printing
#' @param LSitr Integer specifying do <LSitr> local search iterations on parameter values where we do actually do LS; NA does no local search
#' @param LSspc Integer specifying number of hyperparameters to conduct local search: conduct local search every <LSspc>^th iteration. NA does no local search
#' @param maxIter Integer specifying max iterations of coordinate descent
#' @import JuliaConnectoR
#' @import dplyr
#' @importFrom caret createFolds
#' @export

sparseCV_MT <- function(data,
                     tune.grid,
                     hoso = "hoso", # could balancedCV (study balanced CV necessary if K =2)
                     method = "L0", # could be L0 for sparse regression, MS or MS2 (with beta bar penalty)
                     nfolds = "K",
                     juliaFnPath = NA,
                     messageInd = FALSE, # if TRUE then show messages about tuning status
                     LSitr = 50, # do <LSitr> local search iterations on parameter values where we do actually do LS; NA does no local search
                     LSspc = 1, # do local search every <LSspc>^th iteration. NA does no local search
                     maxIter = 2500
                     ){

    # rename studies from 1:K
    XY <- substr(colnames(data), 1, 1) # extract first letter and see which one is Y
    K <- length( which(XY == "Y") )
    Yindx <- 1:K
    Xindx <- seq( K + 1, ncol(data) )
    num.trainStudy <- K 
    WSmethod <- 2 # dummy
    ASpass <- TRUE # dummy
    ASmultiplier <- 3
    method_nm <- method_nm(method = method, multiLabel = TRUE)
    ########################

    ##########################################################
    # order tuning grid correctly for proper warm-starts
    ##########################################################
    colNm <- colnames(tune.grid)
    if ("lambda_z" %in% colNm ){
      tune.grid <- tune.grid[  order(tune.grid$rho, # decreasing rho
                                     -tune.grid$lambda_z, # incraesing lambda_z
                                     tune.grid$lambda1, # decreasing lambda1 (for ridge term)
                                     -tune.grid$lambda2, # increasing lambda2 (for betabar term)
                                     decreasing=TRUE),     ]
    }else{
      if ("lambda1" %in% colNm ){

      # if just rho and lambda (no lambda_z)
      tune.grid <- tune.grid[  order(tune.grid$rho, # decreasing rho
                                     tune.grid$lambda1, # decreasing lambda1 (for ridge term)
                                     -tune.grid$lambda2,
                                             decreasing=TRUE),     ]
      }else{
        # if just rho and lambda (no lambda_z)
        tune.grid <- tune.grid[  order(tune.grid$rho, # decreasing rho
                                       tune.grid$lambda, # decreasing lambda
                                       decreasing=TRUE),     ]
      }
    }


    ########################################
    # create folds depending on hoso type
    ########################################

    if(nfolds == "K")  nfolds <- K

    indxL <- HOOL <- vector(length = nfolds, "list") # lists of indices for training

    if(hoso %in% c("balancedCV", "multiTask") ){
        # balance based on studies
        allRows <- 1:nrow(data)
        set.seed(1)
        HOOL <- createFolds(allRows, k = nfolds) # make folds 
        

        for(study in 1:nfolds){
            Hind <- HOOL[[study]] # indices of study to hold out
            indxL[[study]] <- allRows[-Hind]
            
            # make sure no rho's are bigger than the number of covariates with non-zero variance
            sdVec <- apply( as.matrix( data[ indxL[[study]], Xindx ] ), 2, sd) # standard deviation of features
            sdPos <- sum(sdVec > 0) # number of features with non-zero variance
            
            if( max(tune.grid$rho) > sdPos ){
              # if biggest rho is bigger than number of non-zero variance features
              rhoIndx <- which(tune.grid$rho > sdPos) # indices of rho's which are bigger
              tune.grid$rho[rhoIndx] <- sdPos # set the biggest rho's equal to the most number of features
              
            }
        }
    }
    
    ################################################################################################
    ########################
    # methods
    ########################
    if(method == "L0" & hoso %in% c("balancedCV", "hoso", "multiTask") ){
      # for training "L0" merged model
      if(hoso == "hoso"){
        message(paste0(method_nm, ": Hold-One-Study-Out CV Tuning"))
      }else{
        message(paste0(method_nm, ": ", hoso, " Tuning"))
      } 
        # hold one study out on the merged dataset--used for Merged tuning
        # call merged tuner with corresponding "hoso" version
        mod <-  sMTL::sparseL0Tn_iht(data = data,
                                     tune.grid = tune.grid,
                                     hoso = hoso, # could balancedCV # study balanced CV necessary if K =2
                                     nfolds = nfolds,
                                     juliaPath = juliaPath,
                                     juliaFnPath = juliaFnPath,
                                     trainingStudy = NA, # this is the index of the training study for training each study of an ensemble and testing on all the rest
                                     messageInd = messageInd,
                                     LSitr = LSitr, # do <LSitr> local search iterations on parameter values where we do actually do LS
                                     LSspc = LSspc # do local search every <LSspc>^th iteration
                                    ) 

        return( reName_cv( list(best = mod$best, rmse = mod$rmse) ) )
    ################################################################################################
    }else if(method == "MS" | method == "MS2" | method == "MS_z" | method == "MS_z_fast" | method == "MS_z3" | method == "MS_z_old"){
        # multi study function
      
        # whether sparsity is included is determined by arguments in function
      if(hoso == "hoso"){
        message(paste0(method_nm, ": Hold-One-Study-Out CV"))
      }else{
        message(paste0(method_nm, ": ", hoso, " Tuning"))
      } 
      
        suppressWarnings( rm( L0_MS2, L0_MS, L0_MS_z ) )

          if(method == "MS_z" & !exists("MS_z")){
            L0_MS_z1 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
          }else if(method == "MS_z_fast" & !exists("MS_z_fast")){
            L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest_MT.jl") ) # best for when there is no z- zbar penalty but DOES NOT use active set so studies have same cardinality, same lambda for the ridge penalty, but support can differ. 
          }else if(method == "MS_z3" & !exists("MS_z3") ){
            L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # separate active sets for each study
        }       
        
        if(method == "MS2" & !exists("MS2"))       L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune_MT.jl") ) # \beta - \betaBar penalty
        if(method == "MS" & !exists("MS") )        L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune_MT.jl") ) # Only L2 penalty

        # hold one study out on the merged dataset--used for Merged tuning

        lambdaMat <- matrix(nrow = nfolds, ncol = ncol(tune.grid) ) # stores best lambdas
        rmseMat <- matrix(nrow = nfolds, ncol = nrow(tune.grid) ) # store RMSEs for current study

        ############################
        # HOO CV
        ############################
        # studyVec
        for(fold in 1:nfolds){
            if(messageInd)       message(paste0("Tuning fold: ", fold, ", of ", nfolds, " folds"))

            WS <- TRUE 
            if(fold > 1)    b <- b_init # assign current warm start to warm start from largest cardinality of first fold to warrm start the warm start (so to speak)
          
            indxList <- indxL[[fold]] # indices of studies to train on
            HOOList <- HOOL[[fold]] # indices of study to hold out

            Kfold <- K # number of tasks here (number of betas in dataset)
            # if balanced CV then need to make the warm starts for each fold or if its the first fold need initial warm starts
            # because fold and study do not necessarily match

            ###################################################
            # remove features with no variability in this fold
            ###################################################
            sdVec <- apply( as.matrix(data[indxList, Xindx ]), 2, sd) # standard deviation of features
            
            if( sum(sdVec == 0) > 0 ){
              
              # if some non-zero, remove feature
              IndxRm <- which(sdVec == 0)
              Xindx2 <- Xindx[-IndxRm] # remove feature

            }else{
              
              # do not remove features: use original indices
              Xindx2 <- Xindx

            }
            
            totIndx <- c(Yindx, Xindx2) # indices of dataframe to use for RMSE_MT()
            
            ########################################
            # randomly initialize warm starts
            ########################################
            if(fold == 1){
              b <- matrix( 0, nr = ( length( Xindx2 ) + 1 ), ncol = length(Yindx) ) 
              b_init <- b # initial warm start with largest cardinality for first fold
            } 
            # end warm start
 
            ####################################################
            ############################################################
            # iterate through tuning parameters on current fold
            ############################################################
            rhoVec <- unique(tune.grid$rho)
            
            if( any( rhoVec > length(Xindx2) )  ){
              # if any rho's are bigger than the number of covariates, replace them with number of features (biggest possible)
              
              rhoVec[rhoVec > length(Xindx2)] <- length(Xindx2)
              tune.grid$rho[tune.grid$rho > length(Xindx2)] <- length(Xindx2)
            }
  
            totParams <- length(rhoVec) # total parameters

            for( j in 1:length(rhoVec) ){

                if(messageInd)     message(paste0("Tuning fold: ", fold, ", sparsity (s) ", j, " of ", totParams))


                rhoIndx <- which(tune.grid$rho == rhoVec[j]) # indices of lambda values with this rho value

                if(method == "MS2"){
                  # MS2 -- betabar penalty
                  lambdaVec1 <- tune.grid$lambda1[rhoIndx]
                  lambdaVec2 <- tune.grid$lambda2[rhoIndx]

                  # make sequence of number of local iterations
                  if( is.na(LSitr) | is.na(LSspc) ){
                    # if no local search at all
                    localIter <- rep(0, length(lambdaVec1) )
                  }else{
                    # if local search
                    localIter <- rep(0, length(lambdaVec1) )
                    LSindx <- seq(1, length(lambdaVec1), by = LSspc) # make a sequence thats every LSspc^th
                    localIter[LSindx] <- LSitr # 0 everywhere except every LSspc^th element = LSitr
                    localIter[length(lambdaVec1)] <- LSitr # make last element also have local search so closer to every LSspc^th element
                  }
                  
                  # warm start with greater rho
                  if(WS){
                    b = L0_MS2(X = as.matrix( data[ indxList, Xindx2 ]) ,
                               y = as.matrix(data[ indxList, Yindx ] ),
                               rho = as.integer(min(rhoVec[j] * ASmultiplier, length(Xindx2) - 1)), # warm start with greater rho
                               beta = b,
                               lambda1 = max(lambdaVec1),
                               lambda2 = 0,
                               scale = TRUE,
                               maxIter = as.integer(1000),
                               localIter = as.integer(0))
                    
                    
                    # to warm start the warm start, use the warm start from first fold and largest rho to warm start next fold with largest rho
                    if(j == 1 & fold == 1)    b_init <- b
                  }
                  
                  betas = L0_MS2(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                 y = as.matrix(data[ indxList, Yindx ] ),
                                 rho = as.integer(rhoVec[j]),
                                 beta = as.matrix(b),
                                 lambda1 = lambdaVec1,
                                 lambda2 = lambdaVec2,
                                 scale = TRUE,
                                 maxIter = as.integer(maxIter),
                                 localIter = as.integer(localIter))

                  # save prediction error
                  for(t in 1:length(lambdaVec1) ){
                    # iterate through fits
                    tnIndx <- which(tune.grid$lambda1 == lambdaVec1[t] &
                                      tune.grid$lambda2 == lambdaVec2[t] &
                                        tune.grid$rho == rhoVec[j]
                                    ) # indx of parameter values in tune.grid

                    if(length(lambdaVec1) > 1){
                      betaAvg <- rowMeans( betas[,,t] ) # beta coefficient vector
                    }else{
                      betaAvg <- rowMeans( betas )
                    }

                    # predictions on merged dataset of all held out studies
                    if(hoso == "multiTask"){
                      # if multi task, test on held out studies with multi task RMSE
                      rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = betas[,,t])
                    }  
                    
                  }
                }else if(method == "MS"){
                  # MS -- no betabar penalty
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
                    localIter[length(lambdaVec)] <- LSitr # make last element also have local search so closer to every LSspc^th element
                  }
                  
                  
                  # warm start with greater rho
                  if(WS){
                    b = L0_MS(X = as.matrix( data[ indxList, Xindx2 ]),
                              y = as.matrix(data[ indxList, Yindx ] ),
                              rho = as.integer(min(rhoVec[j] * ASmultiplier, length(Xindx2) - 1)),
                              beta = b,
                              lambda = max(lambdaVec),
                              scale = TRUE,
                              maxIter = as.integer(1000),
                              localIter = as.integer(0))
                    
                     # do WS in next fold
          
                    # to warm start the warm start, use the warm start from first fold and largest rho to warm start next fold with largest rho
                    if(j == 1 & fold == 1)    b_init <- b
                  }
                  
                  
                  betas = L0_MS(X = as.matrix( data[ indxList, Xindx2 ]),
                                y = as.matrix(data[ indxList, Yindx ] ),
                                rho = as.integer(rhoVec[j]),
                                beta = as.matrix(b),
                                lambda = lambdaVec,
                                scale = TRUE,
                                maxIter = as.integer(maxIter),
                                localIter = as.integer(localIter))

                  ## save prediction error
                  for(t in 1:length(lambdaVec) ){
                    # iterate through fits
                    tnIndx <- which(tune.grid$lambda == lambdaVec[t] & tune.grid$rho == rhoVec[j]) # indx of parameter values in tune.grid

                    fitG <- betas[,,t] # beta coefficient vector
                    betaAvg <- rowMeans(fitG) # use average weights (not stacking)

                    # predictions on merged dataset of all held out studies
                    if(hoso == "multiTask"){
                      
                      # if multi task, test on held out studies with multi task RMSE
                      rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = betas[,,t])
                    }  

                  }
                }else if( method == "MS_z"){
                  # MS -- z - zbar penalty (and beta - betaBar penalty)
                  #################################
                  # 1) make sure tehre are lambda_z = 0 ones to start with, maybe just run max lambda_ridge with lambda_z = 0
                  # 2) for l inb lambda_zVec
                  #  #if it is first one make sure it is set to 0 to give warm start -- save this for next value of lambda_z as WS
                  # find all lambda_ridge at this rho and at this value of lambda_z -- get betas
                  #################################
                  lambdaV_Z <- tune.grid$lambda_z[rhoIndx]
                  zVec <- unique( lambdaV_Z  )

                  # ensure a warm start with lambda_z = 0 if it is not in tuning grid
                  lambdaRidge <- max( c(max(tune.grid$lambda1), 1e-3) )  # use highest lambda_ridge term to get warm start
                  lambdaBetaBar <- 0 # set to 0 for warm start since this shrinks betas together when high
                  
                  locIts <- ifelse(is.na(LSitr), 0, LSitr[1] ) # calculate number of local iters for warm start
                  
                  # use lambda_z = 0 solution as a warm start if it is not in tune.grid below
                  if(WS){
                    b <- L0_MS_z1(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                      y = as.matrix(data[ indxList, Yindx ] ),
                                      rho = as.integer(min(rhoVec[j] * ASmultiplier, length(Xindx2) - 1)), # warm start with greater cardinality of support
                                      beta = as.matrix(b),
                                      lambda1 = lambdaRidge,
                                      lambda2 = lambdaBetaBar,
                                      lambda_z = 0,
                                      scale = TRUE,
                                      maxIter = as.integer(maxIter),
                                      localIter = as.integer(0),
                                      WSmethod = as.integer(WSmethod),
                                      ASpass = ASpass)
                    
                    
                    # to warm start the warm start, use the warm start from first fold and largest rho to warm start next fold with largest rho
                    if(j == 1 & fold == 1)    b_init <- b
                  }
                  


                  for(z in zVec){
                    
                    # find lambda_ridge and lambda_{betaBar} that correspond to this level of lambda_z and rho
                    tuneIndx <- which(tune.grid$rho == rhoVec[j] &
                                        tune.grid$lambda_z == z)

                    lambdaVec1 <- tune.grid$lambda1[tuneIndx]
                    lambdaVec2 <- tune.grid$lambda2[tuneIndx]
                    lambdaVecZ <- tune.grid$lambda_z[tuneIndx]
                    
                    # make sequence of number of local iterations
                    if( is.na(LSitr) | is.na(LSspc) ){
                      # if no local search at all
                      localIter <- rep(0, length(lambdaVec1) )
                    }else{
                      # if local search
                      localIter <- rep(0, length(lambdaVec1) )
                      LSindx <- seq(1, length(lambdaVec1), by = LSspc) # make a sequence thats every LSspc^th
                      localIter[LSindx] <- LSitr # 0 everywhere except every LSspc^th element = LSitr
                      localIter[length(lambdaVec1)] <- LSitr # make last element also have local search so closer to every LSspc^th element
                    }
                    
                    betas = L0_MS_z1(X = as.matrix( data[ indxList, Xindx2 ]),
                                    y = as.matrix(data[ indxList, Yindx ]),
                                    rho = as.integer( rhoVec[j] ),
                                    beta = as.matrix(b),
                                    scale = TRUE,
                                    lambda1 = as.vector(lambdaVec1),
                                    lambda2 = as.vector(lambdaVec2),
                                    lambda_z = as.vector(lambdaVecZ),
                                    maxIter = as.integer(maxIter),
                                    localIter = as.integer(localIter),
                                    WSmethod = as.integer(WSmethod),
                                    ASpass = ASpass)

                    if(class(betas)[1] == "array"){
                      #if theres more than one tuning value
                      bStart <- betas[,,1] # use first tuning value (highest lambda_ridge) as warm start for successive lambda_z
                    }else{
                      bStart <- betas
                    }


                    ## save prediction error
                    for(t in 1:length(lambdaVec1) ){
                      # iterate through fits
                      tnIndx <- which(tune.grid$lambda1 == lambdaVec1[t] &
                                        tune.grid$lambda2 == lambdaVec2[t] & # t^th lambda_2 in inner most for loop
                                        tune.grid$lambda_z == z & # zth z in 2nd loop
                                        tune.grid$rho == rhoVec[j] # jth rho in outer most for loop

                      ) # indx of parameter values in tune.grid

                      if(length(lambdaVec1) > 1){
                        fitG <- betas[,,t] # beta coefficient vector
                      }else{
                        fitG <- betas
                      }
                      betaAvg <- rowMeans(fitG) # use average weights (not stacking)

                      # predictions on merged dataset of all held out studies
                      if(hoso == "multiTask"){
                        # if multi task, test on held out studies with multi task RMSE
                        rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = fitG )
                      }  
                      
                    }
                  }
                }else if( method == "MS_z_fast" ){
                  # MS -- z - zbar penalty (and beta - betaBar penalty)
                  #################################
                  # 1) make sure tehre are lambda_z = 0 ones to start with, maybe just run max lambda_ridge with lambda_z = 0
                  # 2) for l inb lambda_zVec
                  #  #if it is first one make sure it is set to 0 to give warm start -- save this for next value of lambda_z as WS
                  # find all lambda_ridge at this rho and at this value of lambda_z -- get betas
                  #################################
                  lambdaV_Z <- tune.grid$lambda_z[rhoIndx]
                  zVec <- unique( lambdaV_Z  )
                  
                  # ensure a warm start with lambda_z = 0 if it is not in tuning grid
                  lambdaRidge <- max( c(max(tune.grid$lambda1), 1e-3) )  # use highest lambda_ridge term to get warm start
                  lambdaBetaBar <- 0 # set to 0 for warm start since this shrinks betas together when high
                  
                  locIts <- ifelse(is.na(LSitr), 0, LSitr[1] ) # calculate number of local iters for warm start
                  
                  # use lambda_z = 0 solution as a warm start if it is not in tune.grid below
                  if(WS){
                    b <- L0_MS_z2(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                 y = as.matrix(data[ indxList, Yindx ] ),
                                 rho = as.integer(min(rhoVec[j] * ASmultiplier, length(Xindx2) - 1)), # warm start with greater cardinality of support
                                 beta = as.matrix(b),
                                 lambda1 = lambdaRidge,
                                 lambda2 = lambdaBetaBar,
                                 lambda_z = 0,
                                 scale = TRUE,
                                 maxIter = as.integer(maxIter),
                                 localIter = as.integer(0),
                                 WSmethod = as.integer(WSmethod),
                                 ASpass = ASpass)
                    
                    
                    # to warm start the warm start, use the warm start from first fold and largest rho to warm start next fold with largest rho
                    if(j == 1 & fold == 1)    b_init <- b
                  }
                  

                  for(z in zVec){
                    
                    # find lambda_ridge and lambda_{betaBar} that correspond to this level of lambda_z and rho
                    tuneIndx <- which(tune.grid$rho == rhoVec[j] &
                                        tune.grid$lambda_z == z)
                    
                    lambdaVec1 <- tune.grid$lambda1[tuneIndx]
                    lambdaVec2 <- tune.grid$lambda2[tuneIndx]
                    lambdaVecZ <- tune.grid$lambda_z[tuneIndx]
                    
                    # make sequence of number of local iterations
                    if( is.na(LSitr) | is.na(LSspc) ){
                      # if no local search at all
                      localIter <- rep(0, length(lambdaVec1) )
                    }else{
                      # if local search
                      localIter <- rep(0, length(lambdaVec1) )
                      LSindx <- seq(1, length(lambdaVec1), by = LSspc) # make a sequence thats every LSspc^th
                      localIter[LSindx] <- LSitr # 0 everywhere except every LSspc^th element = LSitr
                      localIter[length(lambdaVec1)] <- LSitr # make last element also have local search so closer to every LSspc^th element
                    }
                    
                    betas = L0_MS_z2(X = as.matrix( data[ indxList, Xindx2 ]),
                                    y = as.matrix(data[ indxList, Yindx ]),
                                    rho = as.integer( rhoVec[j] ),
                                    beta = as.matrix(b),
                                    scale = TRUE,
                                    lambda1 = as.vector(lambdaVec1),
                                    lambda2 = as.vector(lambdaVec2),
                                    lambda_z = as.vector(lambdaVecZ),
                                    maxIter = as.integer(maxIter),
                                    localIter = as.integer(localIter),
                                    WSmethod = as.integer(WSmethod),
                                    ASpass = ASpass)
                    
                    if(class(betas)[1] == "array"){
                      #if theres more than one tuning value
                      bStart <- betas[,,1] # use first tuning value (highest lambda_ridge) as warm start for successive lambda_z
                    }else{
                      bStart <- betas
                    }
                    
                    
                    ## save prediction error
                    for(t in 1:length(lambdaVec1) ){
                      # iterate through fits
                      tnIndx <- which(tune.grid$lambda1 == lambdaVec1[t] &
                                        tune.grid$lambda2 == lambdaVec2[t] & # t^th lambda_2 in inner most for loop
                                        tune.grid$lambda_z == z & # zth z in 2nd loop
                                        tune.grid$rho == rhoVec[j] # jth rho in outer most for loop
                                      
                      ) # indx of parameter values in tune.grid
                      
                      if(length(lambdaVec1) > 1){
                        fitG <- betas[,,t] # beta coefficient vector
                      }else{
                        fitG <- betas
                      }
                      betaAvg <- rowMeans(fitG) # use average weights (not stacking)
                      
                      # predictions on merged dataset of all held out studies
                      if(hoso == "multiTask"){
                        # if multi task, test on held out studies with multi task RMSE
                        rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = fitG )
                      }  
                      
                    }
                  }
                }else if( method == "MS_z3" ){
                  # MS -- z - zbar penalty (and beta - betaBar penalty)
                  #################################
                  # 1) make sure tehre are lambda_z = 0 ones to start with, maybe just run max lambda_ridge with lambda_z = 0
                  # 2) for l inb lambda_zVec
                  #  #if it is first one make sure it is set to 0 to give warm start -- save this for next value of lambda_z as WS
                  # find all lambda_ridge at this rho and at this value of lambda_z -- get betas
                  #################################
                  lambdaV_Z <- tune.grid$lambda_z[rhoIndx]
                  zVec <- unique( lambdaV_Z  )
                  
                  # ensure a warm start with lambda_z = 0 if it is not in tuning grid
                  lambdaRidge <- max( c(max(tune.grid$lambda1), 1e-3) )  # use highest lambda_ridge term to get warm start
                  lambdaBetaBar <- 0 # set to 0 for warm start since this shrinks betas together when high
                  
                  locIts <- ifelse(is.na(LSitr), 0, LSitr[1] ) # calculate number of local iters for warm start
                  
                  # use lambda_z = 0 solution as a warm start if it is not in tune.grid below
                  if(WS){
                    b <- L0_MS_z3(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                 y = as.matrix(data[ indxList, Yindx ] ),
                                 rho = as.integer(min(rhoVec[j] * ASmultiplier, length(Xindx2) - 1)), # warm start with greater cardinality of support
                                 beta = as.matrix(b),
                                 lambda1 = lambdaRidge,
                                 lambda2 = lambdaBetaBar,
                                 lambda_z = 0,
                                 scale = TRUE,
                                 maxIter = as.integer(maxIter),
                                 localIter = as.integer(0),
                                 WSmethod = as.integer(WSmethod),
                                 ASpass = ASpass)
                    
                    
                    # to warm start the warm start, use the warm start from first fold and largest rho to warm start next fold with largest rho
                    if(j == 1 & fold == 1)    b_init <- b
                  }
                  
                  
                  for(z in zVec){
                    
                    # find lambda_ridge and lambda_{betaBar} that correspond to this level of lambda_z and rho
                    tuneIndx <- which(tune.grid$rho == rhoVec[j] &
                                        tune.grid$lambda_z == z)
                    
                    lambdaVec1 <- tune.grid$lambda1[tuneIndx]
                    lambdaVec2 <- tune.grid$lambda2[tuneIndx]
                    lambdaVecZ <- tune.grid$lambda_z[tuneIndx]
                    
                    # make sequence of number of local iterations
                    if( is.na(LSitr) | is.na(LSspc) ){
                      # if no local search at all
                      localIter <- rep(0, length(lambdaVec1) )
                    }else{
                      # if local search
                      localIter <- rep(0, length(lambdaVec1) )
                      LSindx <- seq(1, length(lambdaVec1), by = LSspc) # make a sequence thats every LSspc^th
                      localIter[LSindx] <- LSitr # 0 everywhere except every LSspc^th element = LSitr
                      localIter[length(lambdaVec1)] <- LSitr # make last element also have local search so closer to every LSspc^th element
                    }
                    
                    betas = L0_MS_z3(X = as.matrix( data[ indxList, Xindx2 ]),
                                    y = as.matrix(data[ indxList, Yindx ]),
                                    rho = as.integer( rhoVec[j] ),
                                    beta = as.matrix(b),
                                    scale = TRUE,
                                    lambda1 = as.vector(lambdaVec1),
                                    lambda2 = as.vector(lambdaVec2),
                                    lambda_z = as.vector(lambdaVecZ),
                                    maxIter = as.integer(maxIter),
                                    localIter = as.integer(localIter),
                                    WSmethod = as.integer(WSmethod),
                                    ASpass = ASpass)
                    
                    if(class(betas)[1] == "array"){
                      #if theres more than one tuning value
                      bStart <- betas[,,1] # use first tuning value (highest lambda_ridge) as warm start for successive lambda_z
                    }else{
                      bStart <- betas
                    }
                    
                    
                    ## save prediction error
                    for(t in 1:length(lambdaVec1) ){
                      # iterate through fits
                      tnIndx <- which(tune.grid$lambda1 == lambdaVec1[t] &
                                        tune.grid$lambda2 == lambdaVec2[t] & # t^th lambda_2 in inner most for loop
                                        tune.grid$lambda_z == z & # zth z in 2nd loop
                                        tune.grid$rho == rhoVec[j] # jth rho in outer most for loop
                                      
                      ) # indx of parameter values in tune.grid
                      
                      if(length(lambdaVec1) > 1){
                        fitG <- betas[,,t] # beta coefficient vector
                      }else{
                        fitG <- betas
                      }
                      betaAvg <- rowMeans(fitG) # use average weights (not stacking)
                      
                      # predictions on merged dataset of all held out studies
                      if(hoso == "multiTask"){
                        # if multi task, test on held out studies with multi task RMSE
                        rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = fitG )
                      }  
                      
                    }
                  }
                }

            }
        }

        rm(betas)

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
        return( reName_cv( list(best = testLambda, best.1se = se1, rmse = rmseMat, avg = avg)) )
    ################################################################################################
    }else if(hoso == "sse" | hoso == "sseOut"){
        # for training ensembles in two different ways of training each study-specific model
      message(paste0(method_nm, ": ", hoso, " Tuning"))
          # "sse" == fit individual studies for stacking with WITHIN study CV
          # "sse" == fit individual studies and test on all other studies


        #lambdas <- matrix(nrow = num.trainStudy, ncol = ncol(tune.grid) ) # has as many columns as there are parameters
        paramMat <- matrix(nrow = num.trainStudy, ncol = ncol(tune.grid) ) # has as many columns as there are parameters
        avgList <- rmseMat <- vector(length = num.trainStudy, "list") # each element is a matrix of tuning
        # iterate through studies and each time do within-study CV
        for(study in 1:num.trainStudy){

            # set hoso type for sparseL0Tn algorithm below and determine which rows to include accordingly
            if(hoso == "sse"){
              hosoType <- "balancedCV" # this tunes within study and makes arbitrary CV ("balanced CV" with one study creates arbitrary folds)
              indxList <- which(data$Study == study) # only use indices of this study because doing within study CV
              studyInd <- NA # set as NA to be safe here since we are only training on this study and not including other studies
            }else if(hoso == "sseOut"){
              hosoType <- "out" # this trains on within study and tests on the the rest of the others merged together
              indxList <- 1:nrow(data) # do all rows because this trains on one study and tests on the rest
              studyInd <- study # tell which study to train on for hoso = "out"
              }

            if(messageInd)    message(paste0("Tuning study: ", study, ", of ", num.trainStudy, " studies"))

            studyCV <- sMTL::sparseL0Tn_iht(data = data[indxList,], # do "merged" on just this study
                                            tune.grid = tune.grid,
                                            hoso = hosoType, # could balancedCV # study balanced CV necessary if K =2
                                            nfolds = nfolds, # do this as cvFolds because here we never do "hoso" we only do 10-fold CV style training
                                            juliaFnPath = juliaFnPath,
                                            trainingStudy = studyInd, # only necessary if hoso = "out"
                                            messageInd = messageInd,
                                            LSitr = LSitr, # do <LSitr> local search iterations on parameter values where we do actually do LS
                                            LSspc =LSspc
                                            )

            paramMat[study,] <- as.numeric( studyCV$best ) # save best parameter into kth row of paramMat matrix
            rmseMat[[study]] <- studyCV$rmse # store RMSE matrix in each element of list
            avgList[[study]] <- studyCV$avg
        }

        paramMat <- as.data.frame(paramMat)
        colnames(paramMat) <- colnames(tune.grid)

        if( length(rhoVec) > 1){
          se1 <- sMTL::seReturn(avg)
        }else{
          se1 <- NA
        }   
        
        return( reName_cv( list(best = paramMat, best.1se = se1, rmse = rmseMat, avg = avgList)) )
        
    }

}

