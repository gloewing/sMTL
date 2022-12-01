########################
# sparse functions
########################
library(JuliaConnectoR)

#######################
# sparse HOSO CV
#######################
# consider using this for sharing Gurobi enviornment:
# https://github.com/jump-dev/Gurobi.jl
sparseCV_iht <- function(data,
                     tune.grid,
                     hoso = "hoso", # could balancedCV (study balanced CV necessary if K =2)
                     method = "L0", # could be L0 for sparse regression, MS or MS2 (with beta bar penalty)
                     nfolds = "K",
                     cvFolds = 10,
                     glPenalty = TRUE, # if TRUE then use group L2 penalty like group lasso,
                     juliaPath = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin",
                     juliaFnPath = "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/",
                     messageInd = FALSE, # if TRUE then show messages about tuning status
                     LSitr = 5, # do <LSitr> local search iterations on parameter values where we do actually do LS; NA does no local search
                     LSspc = 5, # do local search every <LSspc>^th iteration. NA does no local search
                     threads = NA, # just keep this as a dummy argument to make switching between the parallel and non-parllel versions easier
                     WSmethod = 1,
                     ASpass = FALSE, 
                     maxIter = 1000,
                     se = FALSE # if TRUE, use the 1se rule for selecting the "best" parameter values
                     ){

    # IP determines whether "MS" (multi-study) solves convex or IP version
    # cvFolds is standard cross validation
    message("No Parallelalization")
    library(caret)
    library(L0Learn)
    library(JuliaConnectoR)

    # L0 regression
    # L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0reg.jl") ) # sparseReg -- Gurobi
    # L0_MS <- juliaCall("include", paste0(juliaFnPath, "l0_MS.jl") ) # sparseReg -- gurobi

    # rename studies from 1:K
    XY <- substr(colnames(data), 1, 1) # extract first letter and see which one is Y
    K <- length( which(XY == "Y") )
    Yindx <- 1:K
    Xindx <- seq( K + 1, ncol(data) )
    num.trainStudy <- K 
    #data$Study <- as.numeric( as.factor( data$Study) ) # replace with new study labels from 1:K
    # studyVec <- studies <- unique(data$Study)
    ########################

    ##########################################################
    # order tuning grid correctly for proper warm-starts
    ##########################################################
    colNm <- colnames(tune.grid)
    if ("lambda_z" %in% colNm ){
      # if lambda_z, lambda2, lambda1, rho
      tune.grid <- tune.grid[  order(-tune.grid$rho, #increasing rho
                                     -tune.grid$lambda_z, # incraesing lambda_z
                                     tune.grid$lambda1, # decreasing lambda1 (for ridge term)
                                     -tune.grid$lambda2, # increasing lambda2 (for betabar term)
                                     decreasing=TRUE),     ]
    }else{
      if ("lambda1" %in% colNm ){

      # if just rho and lambda (no lambda_z)
      tune.grid <- tune.grid[  order(-tune.grid$rho, # increasing rho
                                     tune.grid$lambda1, # decreasing lambda1 (for ridge term)
                                     -tune.grid$lambda2,
                                             decreasing=TRUE),     ]
      }else{
        # if just rho and lambda (no lambda_z)
        tune.grid <- tune.grid[  order(-tune.grid$rho, # increasing rho
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
        message(paste0(method, ": ", hoso))
        # hold one study out on the merged dataset--used for Merged tuning
        # call merged tuner with corresponding "hoso" version
        mod <-      sparseL0Tn_iht(data = data,
                               tune.grid = tune.grid,
                               hoso = hoso, # could balancedCV # study balanced CV necessary if K =2
                               nfolds = nfolds,
                               cvFolds = cvFolds,
                               juliaPath = juliaPath,
                               juliaFnPath = juliaFnPath,
                               trainingStudy = NA, # this is the index of the training study for training each study of an ensemble and testing on all the rest
                               messageInd = messageInd,
                               LSitr = LSitr, # do <LSitr> local search iterations on parameter values where we do actually do LS
                               LSspc = LSspc, # do local search every <LSspc>^th iteration
                               se = se

          )

        return( list(best = mod$best, rmse = mod$rmse) )
    ################################################################################################
    }else if(method == "MS" | method == "MS2" | method == "MS_z" | method == "MS_z_fast" | method == "MS_z3" | method == "MS_z_old"){
        # multi study function
      
        # whether sparsity is included is determined by arguments in function
        message(paste0(method, ": ", hoso))
      
        Sys.setenv(JULIA_BINDIR = juliaPath)
        suppressWarnings( rm( maxEigen, L0_MS2, L0_MS, L0_MS_z ) )
        
        if(method == "MS_z_old"){
          method <- "MS_z" 
          # not converted to _MT version
          # L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_oldTest.jl") ) # \beta - \betaBar penalty --- BlockComIHT_inexactAS_tune_oldTest.jl is the same as BlockComIHT_inexactAS_tune_old.jl but includes some dummy variables
        }else if(method == "MS_z"){
          #L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tuneTest.jl") ) #"BlockComIHT_inexactAS_tune.jl") ) # z - zbar with active set
          L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
          }else if(method == "MS_z_fast"){
          L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest_MT.jl") ) # best for when there is no z- zbar penalty but DOES NOT use active set so studies have same cardinality, same lambda for the ridge penalty, but support can differ. 
        }else if(method == "MS_z3"){
          L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # separate active sets for each study
        }       
        
        maxEigen <- juliaCall("include", paste0(juliaFnPath, "eigen.jl") ) # max eigenvalue
        if(method == "MS2")       L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune_MT.jl") ) # \beta - \betaBar penalty
        #if(method == "MS_z")      L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune.jl") ) # z - zbar ---- OLD: replaced 10/5/21. used to be: "BlockComIHT_inexact_tune.jl") ) # z - zBar penalty  #
        if(method == "MS")        L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune_MT.jl") ) # Only L2 penalty

        # hold one study out on the merged dataset--used for Merged tuning

        # if(!IP){
        #     # set max support and rho to be number of covaraites if this is not cardinality constrained
        #     tune.grid$rho <- maxSuppSize <- ncol(data) - 2 # if no IP then no cardinality constraint necessary--set all to max
        #   }else{
        #     maxSuppSize <- max(tune.grid$rho)
        # }


        lambdaMat <- matrix(nrow = nfolds, ncol = ncol(tune.grid) ) # stores best lambdas
        rmseMat <- matrix(nrow = nfolds, ncol = nrow(tune.grid) ) # store RMSEs for current study

        ############################
        # HOO CV
        ############################
        # studyVec

        for(fold in 1:nfolds){
            if(messageInd)       message(paste0("Tuning fold: ", fold, ", of ", nfolds, " folds"))

            indxList <- indxL[[fold]] # indices of studies to train on
            HOOList <- HOOL[[fold]] # indices of study to hold out
            # # HOOList <- which(data$Study == study) # indices of study to hold out
            # # indxList <- which(data$Study != study) # indices of studies to train on
            # 
            # studyMat <- matrix(nrow = length(indxList), ncol = 2) # matrix of original study label and study labels within this fold
            # studyMat[,1] <- data$Study[indxList] # original study labeks
            # studyMat[,2] <- as.numeric( as.factor( data$Study[indxList] ) ) # "new" study labels
            # colnames(studyMat) <- c("original", "foldKey")
            # studyKey <- unique(studyMat) # unique matrix with only nfold rows that provides study key for each study label
            # 
            # # vector of study labels for this fold
            # foldStudyVec <- unique( data$Study[indxList] )
            # Kfold <- length(foldStudyVec) # number of unique studies in this training fold
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
              #bStart2 <- bStart[-(IndxRm + 1), ] # remove coefficient (account for intercept) if feature has 0 sd
              
            }else{
              
              # do not remove features: use original indices
              Xindx2 <- Xindx
              #bStart2 <- bStart
              
            }
            
            totIndx <- c(Yindx, Xindx2) # indices of dataframe to use for RMSE_MT()
            
            ########################################
            # randomly initialize warm starts
            ########################################
            set.seed(1) # try to keep initializations fixed
            totalBetas <- ( length( Xindx2 ) + 1 ) * Kfold
            b <- matrix( 0, nr = ( length( Xindx2 ) + 1 ), ncol = length(Yindx) ) #matrix( rnorm(totalBetas), nr = ncol(data) - 1, ncol = Kfold)
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
            
            # max eigenvalue for step size -- for this specific fold
            L <- maxEigen(X = as.matrix( data[ indxList, Xindx2 ]),
                          intercept = TRUE
                          )
            
            # if(method == "MS_z3"){
            #   # calculate max singular values for each study
            #   
            #   eigenVec <- vector( length = Kfold )
            #   
            #   for(ii in 1:Kfold){
            #     
            #     studyNumber <- studyKey[ii, 1] # current study number
            #     studyIndex <- which(data$Study == studyNumber) # which observations are in this study
            #     studyIndex <- intersect(studyIndex, indxList) # which observations are in this fold and for this study
            #     
            #     eigenVec[ii] <- maxEigen(X = as.matrix( data[ studyIndex, -c(1,2) ]),
            #                              #study = NA, # single study so set to NA
            #                              intercept = TRUE
            #                             )
            #     
            #   }
            #     
            # }else{
            #   eigenVec <- NA # use as a dummy for other cases MS_z and MS_z_fast
            # } 

            for( j in 1:length(rhoVec) ){

                if(messageInd)     message(paste0("Tuning fold: ", fold, ", rho ", j, " of ", totParams))


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
                  b = L0_MS2(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                 y = as.matrix(data[ indxList, Yindx ] ),
                                 rho = min(rhoVec[j] * 3, length(Xindx2) - 1), # warm start with greater rho
                                 # study = as.vector( studyMat[,2] ), # these are the study labels ordered appropriately for this fold
                                 beta = b,
                                 lambda1 = max(lambdaVec1),
                                 lambda2 = 0,
                                 scale = TRUE,
                                 maxIter = maxIter,
                                 #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                 localIter = 0)
                  
                  betas = L0_MS2(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                 y = as.matrix(data[ indxList, Yindx ] ),
                                 rho = rhoVec[j],
                                 #study = as.vector( studyMat[,2] ), # these are the study labels ordered appropriately for this fold
                                 beta = as.matrix(b),
                                 lambda1 = lambdaVec1,
                                 lambda2 = lambdaVec2,
                                 scale = TRUE,
                                 maxIter = maxIter,
                                  #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                 localIter = localIter)

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
                    #fitG <- betas[,,t] # beta coefficient vector

                    # predictions on merged dataset of all held out studies
                    # preds <- as.vector(  as.matrix( cbind(1, data[HOOList, Xindx]) ) %*% betaAvg   )
                    # rmseMat[fold, tnIndx]  <- sqrt(mean( (preds - data$Y[HOOList] )^2  )) # rmse
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
                  b = L0_MS(X = as.matrix( data[ indxList, Xindx2 ]),
                                y = as.matrix(data[ indxList, Yindx ] ),
                                rho = min(rhoVec[j] * 3, length(Xindx2) - 1),
                                # study = as.vector( studyMat[,2] ), # these are the study labels ordered appropriately for this fold
                                beta = b,
                                lambda = max(lambdaVec),
                                scale = TRUE,
                                maxIter = maxIter,
                                #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                localIter = 0)
                  
                  betas = L0_MS(X = as.matrix( data[ indxList, Xindx2 ]),
                                y = as.matrix(data[ indxList, Yindx ] ),
                                rho = rhoVec[j],
                                # study = as.vector( studyMat[,2] ), # these are the study labels ordered appropriately for this fold
                                beta = as.matrix(b),
                                lambda = lambdaVec,
                                scale = TRUE,
                                maxIter = maxIter,
                                 #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                localIter = localIter)

                  ## save prediction error
                  for(t in 1:length(lambdaVec) ){
                    # iterate through fits
                    tnIndx <- which(tune.grid$lambda == lambdaVec[t] & tune.grid$rho == rhoVec[j]) # indx of parameter values in tune.grid

                    fitG <- betas[,,t] # beta coefficient vector
                    betaAvg <- rowMeans(fitG) # use average weights (not stacking)

                    # predictions on merged dataset of all held out studies
                    # preds <- as.vector(  as.matrix( cbind(1, data[HOOList, Xindx]) ) %*% betaAvg   )
                    # rmseMat[fold, tnIndx]  <- sqrt(mean( (preds - data$Y[HOOList] )^2  )) # rmse
                    
                    if(hoso == "multiTask"){
                      
                      # if multi task, test on held out studies with multi task RMSE
                      rmseMat[fold, tnIndx]  <- multiTaskRmse_MT(data = data[HOOList, totIndx], beta = betas[,,t])
                    }  

                  }
                }else if( method == "MS_z" | method == "MS_z_fast" | method == "MS_z3" ){
                  # MS -- z - zbar penalty (and beta - betaBar penalty)
                  #################################
                  # 1) make sure tehre are lambda_z = 0 ones to start with, maybe just run max lambda_ridge with lambda_z = 0
                  # 2) for l inb lambda_zVec
                  #  #if it is first one make sure it is set to 0 to give warm start -- save this for next value of lambda_z as WS
                  # find all lambda_ridge at this rho and at this value of lambda_z -- get betas
                  #################################
                  lambdaV_Z <- tune.grid$lambda_z[rhoIndx]
                  zVec <- unique( lambdaV_Z  )

                  #if(!0 %in% zVec){
                  # ensure a warm start with lambda_z = 0 if it is not in tuning grid
                  lambdaRidge <- max( c(max(tune.grid$lambda1), 1e-3) )  # use highest lambda_ridge term to get warm start
                  lambdaBetaBar <- 0 # set to 0 for warm start since this shrinks betas together when high
                  
                  locIts <- ifelse(is.na(LSitr), 0, LSitr[1] ) # calculate number of local iters for warm start
                  
                  # use lambda_z = 0 solution as a warm start if it is not in tune.grid below
                  bStart <- L0_MS_z(X = as.matrix( data[ indxList, Xindx2 ]) ,
                                    y = as.matrix(data[ indxList, Yindx ] ),
                                    rho = min(rhoVec[j] * 3, length(Xindx2) - 1), # warm start with greater cardinality of support
                                    # study = as.vector( studyMat[,2] ), # these are the study labels ordered appropriately for this fold
                                    beta = as.matrix(b),
                                    lambda1 = lambdaRidge,
                                    lambda2 = lambdaBetaBar,
                                    lambda_z = 0,
                                    scale = TRUE,
                                    maxIter = maxIter,
                                    #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                    localIter = 0,
                                    WSmethod = WSmethod,
                                    ASpass = ASpass)
                    
                  # }else{
                  #   bStart <- b
                  # }

                  for(z in zVec){
                    
                    # print(z)
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
                    
                    betas = L0_MS_z(X = as.matrix( data[ indxList, Xindx2 ]),
                                    y = as.matrix(data[ indxList, Yindx ]),
                                    rho = as.integer( rhoVec[j] ),
                                    # study = as.vector( studyMat[,2] ), 
                                    beta = as.matrix(bStart),
                                    scale = TRUE,
                                    lambda1 = as.vector(lambdaVec1),
                                    lambda2 = as.vector(lambdaVec2),
                                    lambda_z = as.vector(lambdaVecZ),
                                    maxIter = maxIter,
                                    localIter = localIter,
                                    # eigenVec = eigenVec,
                                     #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                                    WSmethod = WSmethod,
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
                      # preds <- as.vector(  as.matrix( cbind(1, data[HOOList, Xindx]) ) %*% betaAvg   )
                      # rmseMat[fold, tnIndx]  <- sqrt(mean( (preds - data$Y[HOOList] )^2  )) # rmse

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
        if(se)     testLambda <- seReturn(errorMean) # 1se rule
        colnames(testLambda) <- colnames(tune.grid)
        rownames(rmseMat) <- paste0("fold_", 1:nrow(rmseMat) ) # put before line below: rbind( t(tune.grid), rmseMat)
        avg <- colMeans( rmseMat )

        rmseMat <- rbind( t(tune.grid), rmseMat)
        avg <- rbind( t(tune.grid), avg)
        colnames(rmseMat) <- colnames(avg) <- paste0("param", 1:nrow(tune.grid))

        return(list(best = testLambda, rmse = rmseMat, avg = avg))
    ################################################################################################
    }else if(hoso == "sse" | hoso == "sseOut"){
        # for training ensembles in two different ways of training each study-specific model
          message(paste0(method, ": ", hoso ) )
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

            studyCV <- sparseL0Tn_iht(data = data[indxList,], # do "merged" on just this study
                                  tune.grid = tune.grid,
                                  hoso = hosoType, # could balancedCV # study balanced CV necessary if K =2
                                  nfolds = cvFolds, # do this as cvFolds because here we never do "hoso" we only do 10-fold CV style training
                                  cvFolds = cvFolds,
                                  juliaPath = juliaPath,
                                  juliaFnPath = juliaFnPath,
                                  trainingStudy = studyInd, # only necessary if hoso = "out"
                                  messageInd = messageInd,
                                  LSitr = LSitr, # do <LSitr> local search iterations on parameter values where we do actually do LS
                                  LSspc =LSspc,
                                  se = se
                                  )

            paramMat[study,] <- as.numeric( studyCV$best ) # save best parameter into kth row of paramMat matrix
            rmseMat[[study]] <- studyCV$rmse # store RMSE matrix in each element of list
            avgList[[study]] <- studyCV$avg
        }

        paramMat <- as.data.frame(paramMat)
        colnames(paramMat) <- colnames(tune.grid)

        return( list( best = paramMat, rmse = rmseMat, avg = avgList )  )

    }

}



#######################
# sparse HOSO CV
#######################
# this function is called upon by sparseCV above and does L0 tuning with L2 penalty with various
# ways of generating train and validation sets
sparseL0Tn_iht <- function(data,
                     tune.grid,
                     hoso = "hoso", # could balancedCV # study balanced CV necessary if K =2
                     nfolds = "K",
                     cvFolds = 10,
                     IP = TRUE,
                     timeLimit = 180, # 3 minute time limit for tuning
                     juliaPath = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin",
                     juliaFnPath = "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/",
                     trainingStudy = NA, # this is the index of the training study for training each study of an ensemble and testing on all the rest
                     # trainingStudy index is only called on internally when cvOUT internally calls on method = "merged", hoso = "out"
                     messageInd = FALSE, # if true then show messages about tuning status
                     LSitr = 5, # do <LSitr> local search iterations on parameter values where we do actually do LS
                     LSspc = 5, # do local search every <LSspc>^th iteration
                     maxIter = 1000,
                     se = FALSE # if TRUE then use 1se rule for selecting rho
                     ){

  # IP determines whether "MS" (multi-study) solves convex or IP version
  # cvFolds is standard cross validation

  library(caret)
  library(L0Learn)
  library(JuliaConnectoR)

  # read in Julia
  Sys.setenv(JULIA_BINDIR = juliaPath)

  # L0 regression
  suppressWarnings( rm( maxEigen, L0_reg ) )
  
  maxEigen <- juliaCall("include", paste0(juliaFnPath, "eigen.jl") ) # max eigenvalue
  L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # IHT

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
    HOOL <- createFolds(factor(data$Study), k = nfolds) # make folds with balanced studies
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
    b <- rep(0, ncol(data) - 1) #rnorm( ncol(data) - 1 )

    for(study in 1:nfolds){

      if(messageInd)    message(paste0("Tuning fold: ", study, ", of ", nfolds, " folds"))

      indxList <- indxL[[study]] # indices of studies to train on
      HOOList <- HOOL[[study]] # indices of study to hold out

      nk <- length(indxList) # sample size of kth study
      rhoVec <- unique(tune.grid$rho)

      ##########################
      # HOO CV
      ##########################
      totParams <- nrow(tune.grid) # total parameters
      # max eigenvalue for Lipschitz constant
      L <- maxEigen(X = as.matrix( data[ indxList, Xindx ]),
                    intercept = TRUE
                    )

      for(j in 1:length(rhoVec)){

        if(messageInd)    message(paste0("Tuning fold: ", study, ", parameter ", j, " of ", totParams))

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
        b = L0_reg(X = as.matrix(data[ indxList, Xindx  ]),
                     y = data$Y[  indxList  ],
                     rho = min( rhoVec[j], length(Xindx) - 1),
                     beta = b,
                     lambda = max(lambdaVec),
                     scale = TRUE,
                     maxIter = maxIter,
                     #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                     localIter = 0
                    )
        
        # L0 constrained ridge estimator
        fit = L0_reg(X = as.matrix(data[ indxList, Xindx  ]),
                      y = data$Y[  indxList  ],
                      rho = rhoVec[j],
                      beta = as.matrix(b),
                      lambda = lambdaVec,
                      scale = TRUE,
                      maxIter = maxIter,
                       #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                     localIter = localIter
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
    if(se)     testLambda <- seReturn(errorMean) # 1se rule
    
    colnames(testLambda) <- colnames(tune.grid)
    rownames(rmseMat) <- paste0("fold_", 1:nrow(rmseMat) ) # put before line below: rbind( t(tune.grid), rmseMat)
    avg <- colMeans( rmseMat )

    rmseMat <- rbind( t(tune.grid), rmseMat)
    avg <- rbind( t(tune.grid), avg)
    colnames(rmseMat) <- colnames(avg) <- paste0("param", 1:nrow(tune.grid))


    return( list(best = testLambda, rmse = rmseMat, avg = avg) )

}


sparseL0Tn_iht_par <- function(data,
                           tune.grid,
                           hoso = "hoso", # could balancedCV # study balanced CV necessary if K =2
                           nfolds = "K",
                           cvFolds = 10,
                           IP = TRUE,
                           timeLimit = 180, # 3 minute time limit for tuning
                           juliaPath = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin",
                           juliaFnPath = "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/",
                           trainingStudy = NA, # this is the index of the training study for training each study of an ensemble and testing on all the rest
                           # trainingStudy index is only called on internally when cvOUT internally calls on method = "merged", hoso = "out"
                           messageInd = FALSE, # if true then show messages about tuning status
                           LSitr = 5, # do <LSitr> local search iterations on parameter values where we do actually do LS
                           LSspc = 5, # do local search every <LSspc>^th iteration
                           threads = 5, 
                           maxIter = 1000,
                           se = FALSE # 1se rule for calculating best rho
){
  # same as sparseL0Tn_iht but parallelizes
  # IP determines whether "MS" (multi-study) solves convex or IP version
  # cvFolds is standard cross validation

  library(caret)
  library(glmnet)
  library(L0Learn)
  library(JuliaConnectoR)
  library(doParallel)
  library(foreach)

  # # read in Julia
  # Sys.setenv(JULIA_BINDIR = juliaPath)
  #
  # # L0 regression
  # # L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0reg.jl") ) # sparseReg -- Gurobi
  # # L0_MS <- juliaCall("include", paste0(juliaFnPath, "l0_MS.jl") ) # sparseReg -- gurobi
  # maxEigen <- juliaCall("include", paste0(juliaFnPath, "eigen.jl") ) # max eigenvalue
  # L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # IHT
  # L0_MS_pnlty <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune.jl") ) # \beta - \betaBar penalty
  # L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune.jl") ) # Only L2 penalty


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
    HOOL <- createFolds(factor(data$Study), k = nfolds) # make folds with balanced studies
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
  b <- rep(0, ncol(data) - 1) #rnorm( ncol(data) - 1 )

  # parallelization
  num.threads <- as.integer( min(c(threads, nfolds)) ) # number of threads
  threads <- makeCluster(num.threads, outfile=logfile)
  registerDoParallel(threads)

  rmseMat_par <- foreach(study = 1:nfolds, .combine = rbind, .multicombine = TRUE) %dopar%{


    # read in Julia
    Sys.setenv(JULIA_BINDIR = juliaPath)
    suppressWarnings( rm( maxEigen, L0_reg ) )
    
    maxEigen <- juliaCall("include", paste0(juliaFnPath, "eigen.jl") ) # max eigenvalue
    L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # IHT


    if(messageInd)    message(paste0("Tuning fold: ", study, ", of ", nfolds, " folds"))

    indxList <- indxL[[study]] # indices of studies to train on
    HOOList <- HOOL[[study]] # indices of study to hold out

    nk <- length(indxList) # sample size of kth study
    rhoVec <- unique(tune.grid$rho)

    ##########################
    # HOO CV
    ##########################
    totParams <- nrow(tune.grid) # total parameters
    # max eigenvalue for Lipschitz constant
    L <- maxEigen(X = as.matrix( data[ indxList, Xindx ]),
                  intercept = TRUE
    )

    for(j in 1:length(rhoVec)){

      if(messageInd)    message(paste0("Tuning fold: ", study, ", parameter ", j, " of ", totParams))

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
      b = L0_reg(X = as.matrix(data[ indxList, Xindx  ]),
                 y = data$Y[  indxList  ],
                 rho = min( rhoVec[j], length(Xindx) - 1),
                 beta = b,
                 lambda = max(lambdaVec),
                 scale = TRUE,
                 maxIter = maxIter,
                 #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                 localIter = 0
      )
      
      # L0 constrained ridge estimator
      fit = L0_reg(X = as.matrix(data[ indxList, Xindx  ]),
                   y = data$Y[  indxList  ],
                   rho = rhoVec[j],
                   beta = b,
                   lambda = lambdaVec,
                   scale = TRUE,
                   maxIter = maxIter,
                    #  eig = L, # commented out 3-14-22 because doesn't correspond with study-specific scaled design matrix eigenvales
                   localIter = localIter
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
    return( rmseMat[study, ] )
  }

  rmseMat <- rmseMat_par # set output of foreach loop

  rm(fit, preds, rmseMat_par)

  errorMean <- colMeans(rmseMat)
  testLambda <- tune.grid[ which.min(errorMean),]
  if(se)     testLambda <- seReturn(errorMean) # 1se rule
  
  testLambda <- as.data.frame(testLambda)
  colnames(testLambda) <- colnames(tune.grid)
  rownames(rmseMat) <- paste0("fold_", 1:nrow(rmseMat) ) # put before line below: rbind( t(tune.grid), rmseMat)
  avg <- colMeans( rmseMat )

  rmseMat <- rbind( t(tune.grid), rmseMat)
  avg <- rbind( t(tune.grid), avg)
  colnames(rmseMat) <- colnames(avg) <- paste0("param", 1:nrow(tune.grid))


  return( list(best = testLambda, rmse = rmseMat, avg = avg) )

}

# support statistics like FPR, TPR, FULL SUPPORT RECOVERY, AUC
suppStat <- function(response, predictor){

  # library(pROC)

  # assume false is = 0, and true = 1
  # false positive rate
  totalNeg <- sum(response == 0) # total negatives
  falsePos <- sum(response == 0 & predictor == 1 )
  fp <- falsePos / totalNeg

  # true positive rate
  totalPos <- sum(response == 1) # total positives
  truePos <- sum(response == 1 & predictor == 1 )
  tp <- truePos / totalPos

  # support is completely right
  rightSupport <- all(response == predictor) * 1

  auc <- pROC::auc(response = response, predictor = predictor)

  return( c(fp,
            tp,
            rightSupport,
             auc)
          )
}


saveFn <- function(file, fileNm, iterNum, save.folder = NA){
  
  if( !is.na(save.folder) ){
    # set working directory if specified
    #setwd(save.folder)
    fileNm <- paste0(save.folder, "/", fileNm)
  }
  
  # check if file exists
  if(  file.exists(fileNm)  ){
    # if exists read in file and save this result to correspond to row
    res <- read.csv( fileNm )
    res[iterNum,] <- file[iterNum,]
    write.csv(res, fileNm, row.names = FALSE)
    
  }else{
    # if it does not exist (first iteration to complete) then save resMat
    write.csv(file, fileNm, row.names = FALSE)
  }
  
}


multiTaskSplit <- function(data, split = 0.5){
  K <- length( unique( data$Study ) ) # number of studies
  trainSet <- vector(length = K, mode = "list") # list for indices

  for(j in 1:K){
    indx <- which(data$Study == j) # indices of current studies
    trainIndx <- sample.int( length(indx),  # sample w/o replacement as many studies as there are
                             size = round( length(indx) * split ),
                             replace = FALSE
                             )
    
    trainSet[[j]] <- indx[trainIndx] # add indices to list

  }
  
  trainIndx <- do.call(c, trainSet) # concatenate indices
  
  return( list(train = data[trainIndx, ], test = data[-trainIndx, ] ) )
  
}


# multi task with same design matrix ( just a train/test split function )
multiTaskSplit_MT <- function(data, 
                              split = 0.5){
  #K <- length( unique( data$Study ) ) # number of studies
  trainIndx <- sample.int( nrow(data),  # sample w/o replacement as many studies as there are
                           size = round( nrow(data) * split ),
                           replace = FALSE
  )
  
  return( list(train = data[trainIndx, ], test = data[-trainIndx, ] ) )
  
}


# multi task with same design matrix ( just a train/test split function ) but uses set splots for CV
multiTaskSplit_CV_MT <- function(data, 
                                 seed = 1,
                                 testIndex = 1, # index of fold to save as test dataset
                                  folds = 10){
  
  set.seed(seed)
  library(caret)
  
  # create k folds (disjoint sets)
  dataPart <- createFolds(1:nrow(data), 
                          k = folds)
  testIndx <- as.vector( dataPart[[testIndex]] )# test fold

  
  return( list(train = data[-testIndx, ], test = data[testIndx, ] ) )
  
}


multiTaskRmse <- function(data, beta){
  K <- length( unique( data$Study ) ) # number of studies
  trainSet <- vector(length = K) # list for indices
  
  for(j in 1:K){
    indx <- which(data$Study == j) # indices of current studies
    
    # rmse for jth study using jth beta (i.e., jth model)
    trainSet[j] <- sqrt( mean( 
                          (  data$Y[indx] - cbind(1, as.matrix(data[indx, Xindx]) ) %*% beta[,j]  )^2 
                          ) 
                         )
    
  }
  
  
  return( mean(trainSet) )
  
}

# multi task RMSE for multi-task (same design matrix)
multiTaskRmse_MT <- function(data, 
                             K = NA,
                             beta){
  
  if(is.na(K)){
    XY <- substr(colnames(data), 1, 1) # extract first letter and see which one is Y
    Yindx <- which(XY == "Y") 
    K <- length( Yindx )
    
  }
  
  trainSet <- vector(length = K) # list for indices
  
  for(j in 1:K){
    
    outcomeIndx <- Yindx[j]
    
    # rmse for jth study using jth beta (i.e., jth model)
    trainSet[j] <- sqrt( mean( 
      (  data[, outcomeIndx ] - cbind(1, as.matrix( data[, -seq(1, K) ] ) ) %*% beta[,j]  )^2 
    ) 
    )
    
  }
  
  
  return( mean(trainSet) )
  
}



# multi task R^2 for multi-task (same design matrix)
multiTaskR2_MT <- function(data, Ytrain,
                             K = NA,
                             beta){
  
  if(is.na(K)){
    XY <- substr(colnames(data), 1, 1) # extract first letter and see which one is Y
    Yindx <- which(XY == "Y") 
    K <- length( Yindx )
    
  }
  
  Ymean <- colMeans(Ytrain) # use means from training set
  
  trainSet <- vector(length = K) # list for indices
  
  for(j in 1:K){
    
    outcomeIndx <- Yindx[j]
    
    preds <- cbind(1, as.matrix( data[, -seq(1, K) ] ) ) %*% beta[,j]
    
    # rmse for jth study using jth beta (i.e., jth model)
    trainSet[j] <- 1 - sum( 
                          (  data[, outcomeIndx ] - preds  )^2 
                        ) / sum( 
                          (  data[, outcomeIndx ] - Ymean[j]   )^2 
                        ) 
    
    
  }
  
  print(trainSet)
  
  return( mean(trainSet) )
  
}


# finds scaling factors forr lambda_z based on how big penalty can be
rhoScale <- function(K, p, rhoVec, 
                     itrs = 25000,
                     seed = 1){
  
  # K - number of tasks
  # p - number of covaraites
  # rhoVec is vector of possible rhos
  # is number of random samples
  
  rhoVec <- unique( rhoVec[rhoVec <= p] )# remove any rhos > p
  resMat <- data.frame( matrix(0, nrow = length(rhoVec), ncol = 2) ) # save results
  colnames(resMat) <- c("rho", "scale")
  resMat[,1] <- rhoVec # first column are rhos
  
  for(i in 1:length(rhoVec) ){
    s <- rhoVec[i]
    
    if(s * K <= p){
      
      # exact solution
      resMat[i, 2] <- 2 * choose(K, 2) * s / K
      
    }else{
      # simulate
      
      set.seed(seed)
      resVec <- vector(length = itrs)
      
      vec <- c( rep(0, p - s), rep(1, s) )
      m <- matrix(0, ncol = K, nrow = p)
      
      timeStart <- Sys.time()
      for(itr in 1:itrs){

        # simulate draws
        for(j in 1:K){
          m[,j] <- sample( vec, replace = FALSE )
        }
        
        # calcualte distance
        resVec[itr] <- sum( dist( t(m) )^2 ) / 4
        
      }
      
      resMat[i, 2] <- max(resVec) # take empirical maximum as approximation to true max
    }
    
  }
  
  return(resMat)
}


# takes in tune grid and the output of function rhoScale and re-scales the lambda_z
tuneZscale <- function(tune.grid, rhoScale){
  
  rhoVec <- unique(tune.grid$rho) # unique rows
  
  for(rho in rhoVec ){
    
    indx <- which(tune.grid$rho == rho) # indices of tune.grid with current rho value
    zScale <- rhoScale$scale[rhoScale$rho == rho] # find scaling factor associated with this rho in 

    tune.grid$lambda_z[indx] <- tune.grid$lambda_z[indx] / zScale
    
  }
  
  return(tune.grid)
  
}


# 1se rule for selecting rho

seReturn <- function(x){
  
  # pre-process
  e <- x$avg %>% 
    as_tibble() %>% 
    t() 
  
  colnames(e) <- rownames(x$avg) # new names because we transposed
  
  minVec <- e %>% 
    as_tibble() %>% 
    group_by(rho) %>% 
    summarise(m = min(avg)) # get the minimum for each rho
  
  sdVal <- sd(minVec$m) # get standard deviation across minima
  
  mindx <- which(minVec$m <= min(minVec$m) + sdVal )[1] # get smallest rho that is within rrange
  rhoStar <- minVec$rho[mindx] # index of  best tuning values for this rho
  
  bestVal <- e %>% 
    as_tibble() %>% 
    filter(rho == rhoStar) %>% 
    slice_min(avg) %>%
    select( -avg ) %>%
    as.data.frame()
  
  return(bestVal)
  
}

# support statistics like FPR, TPR, FULL SUPPORT RECOVERY, AUC
suppHet <- function(mat, intercept = TRUE){
  
  mat <- I(mat != 0) * 1 # make indicator matrix
  if(intercept) mat <- mat[-1,] # remove intercept
  
  p <- nrow(mat)
  K <- ncol(mat)
  rho <- mean( colSums(mat) ) # cardinality, each column sum should be the same
  
  # average of all pairwise combinations of columns
  e <- 0
  totalProb <- 0
  for(j in 2:K){
    for(l in 1:(j-1)){
      totalPairs <- mat[,l] %*% mat[,j]
      e <- e + totalPairs
      totalProb <- totalProb + dhyper(totalPairs, p, rho, rho)
      
    }
  }
  
  avgPair <- e / (rho * choose(K, 2))
  totalProb <- totalProb / choose(K, 2) # average probabilities
  
  
  
  
  return( c(avgPair,
            totalProb)
  )
}


