library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)
library(car)

source("sparseFn_iht_test_MT.R") 

cluserInd <- TRUE # whether running on computer or on cluster

save.folder <- "/n/home12/gloewinger/cf_MT7"
load.folder <- "~/Desktop/Research"
if(cluserInd){
    # if on cluster
    args = commandArgs(TRUE)
    n_k <- as.integer( as.numeric(args[1]) ) # sample size to subsample
    colSubSamp <- as.integer( as.numeric(args[2]) )
    iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id
    
    # Julia paths
    juliaPath <- "/n/sw/eb/apps/centos7/Julia/1.5.3-linux-x86_64/bin"
    juliaFnPath <- "/n/home12/gloewinger/" 
}else{
    runNum <- 1
    setwd("~/Desktop/Research")
    iterNum <- 6
    n_k <- 500
    colSubSamp <- 10 # sample every colSubSamp^th row (so = 1 means use all of them and 2 samples every other)
    
    # Julia paths
    juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
    juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune/"
    
}

Sys.setenv(JULIA_BINDIR = juliaPath)

# simulation parameters
K <- totalStudies <- 4
K_train <- K # number of training studies to randomly select

#######################
Yscale <- TRUE # scale outcome

totalSims <- 100
scaleInd <- TRUE

outcome <- "DA" # which is used as an outcome
seedFixedInd <- TRUE # fixed effects (true betas) and Sigma_x fixed across simulation iterations
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- NA #50 #50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- NA #1 #1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 0# 50 # 0 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
rdgLmd <- TRUE # use the ridge hyperparamter from oseL0 to tune the ps2 and ps5 
lambdaZmax <- 5 # make sure lambda_z are smaller than this to pervent numerical instability
dataPro <- 2
Yscale <- TRUE
maxIter_train <- 5000
maxIter_cv <- 1000

tuneInd <- TRUE  #sims6$tuneInd[runNum]
WSmethod <- 2 # sims6$WSmethod[runNum]
ASpass <- TRUE # sims6$ASpass[runNum]

if(tuneThreads == 1){
    # use non-parallel version
    source("sparseFn_iht_test.R") # USE TEST VERSION HERE
    #     source("sparseFn_iht_test_MT.R") # USE TEST VERSION HERE
    sparseCV_iht_par <- sparseCV_iht
}else{
    # source("sparseFn_iht_par.R")
    source("sparseFn_iht_test.R") # USE TEST VERSION HERE
    #     source("sparseFn_iht_test_MT.R") # USE TEST VERSION HERE
    sparseCV_iht_par <- sparseCV_iht
}

# model tuning parameters
L0TuneInd <- TRUE # whether to retune lambda and rho with gurobi OSE (if FALSE then use L0Learn parameters)
L0MrgTuneInd <- TRUE # whether to retune lambda and rho with gurobi Mrg (if FALSE then use L0Learn parameters)
L0_sseTn <- "sse" # tuning for L0 OSE (with gurobi not L0Learn)
MSTn <- "multiTask" #sims6$multiTask[runNum] #"hoso" #"balancedCV" # tuning for MS (could be "hoso")

if(MSTn %in% c("hoso", "balancedCV") ){
    # if not multi-task then this can't be smaller than K
    nfold <- min( 5, K) # 5 fold maximum
}else if(MSTn == "multiTask"){
    # if multi-task then can do 5 fold CV since we are not doing a hold-one-study-out CV
    nfold <- 5
}

p <- 1000 / colSubSamp

rho <- rhoVec <- c(5, 10, 25, 50, 100, 200, 500)
rhoVec <- rhoVec[rhoVec < p]

# rho <- rho[rho <= round(2500 / colSubSamp) ]

rhoDiff <- round( mean( diff(rho) ) / 2 ) # average interval

timeStart <- Sys.time()
#######################
scaleInd <- TRUE

# model tuning parameters
L0TuneInd <- TRUE # whether to retune lambda and rho with gurobi OSE (if FALSE then use L0Learn parameters)
L0MrgTuneInd <- TRUE # whether to retune lambda and rho with gurobi Mrg (if FALSE then use L0Learn parameters)
L0_sseTn <- "sse" # tuning for L0 OSE (with gurobi not L0Learn)
MSTn <- "multiTask" #"balancedCV" # tuning for MS (could be "hoso")
nfoldL0_ose <- nfold <- 5

lamb <- 0.5

lambdaBeta <- c( 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 )

lambda <- sort( unique( c(1e-6, 1e-5, 0.0001, 0.001, 0.01, 5,10, 50, 100, 200, 
                          exp(-seq(0,5, length = 15))
                            ) ), decreasing = TRUE ) 

lambdaShort <- sort( unique( c(1e-4,
                               exp(-seq(0,5, length = 5)),
                               5,10, 50, 100, 250, 500, 1000) ),
                     
                     decreasing = TRUE ) 
lambdaZ <- sort( unique( c(1e-4, 1e-6, 1e-5, 1e-4, 1e-3,
                           exp(-seq(0,5, length = 8)),
                           1:3) ), 
                 decreasing = TRUE )

# file name
fileNm <- paste0("cfFSCV_MT_supp_", 
                 "_datPro_",  dataPro, 
                 "_sclX_", scaleInd,
                 "_rhoLen_", length(rho),
                 "_totSims_", totalSims,
                 "_L0sseTn_", L0_sseTn,
                 "_rdgLmd_", rdgLmd,
                 "_MSTn_", MSTn,
                 "_Yscl_", Yscale,
                 "_rhoMax_", max(rho),
                 "_nFld_", nfold,
                 "_LSitr_", LSitr,
                 "_LSspc_", LSspc,
                 "_fitLocal_", localIters,
                 "_Zlen_", length(lambdaZ),
                 "_wsMeth_", WSmethod,
                 "_asPass_", ASpass,
                 "_TnIn_", tuneInd,
                 "_colSub_", colSubSamp,
                 "_n_k_", n_k, 
                 "_K_", K)

print(fileNm)

timeStart <- Sys.time()


#####################################################################
print(paste0("start: ", iterNum))

Sys.setenv(JULIA_BINDIR = juliaPath)
L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # sparseReg # MT: doesnt make sense
L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune.jl") ) # MT: Need to check it works
L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study

methodNms <- paste0("MT_",
                    c( "mtLasso", "oseL0", "msP1_L0", "msP2_L0", "msP3_L0", "msP5_L0", "msP4_L0")
)
nmLen <- length(methodNms)

methodNms <- paste0(methodNms, "_", rep(1:length(rho), each = nmLen) )
methodNms <- c(methodNms,
               paste0(rep(methodNms, each = 2), rep(c("_pair", "_prob"), nmLen))
)

fullNms <- c(methodNms,
             "MTL_L2L1_supp", "MTL_L2L1", "MTL_trace_supp",
             "MTL_trace", 
             ##
             "mtl_ridge_r2", "MoM_mrg_rho",
             ##
             "MT_lasso_low", "MT_lasso_low_s",
             ##
             "MT_rmtl_lasso_low", "MT_rmtl_lasso_low_s", 
             ##
             "msP1_con",  "msP3_con",
             "time1", "time2", "time3", "totalTime", "timeRidge"
)

resMat <- matrix(nrow = totalSims, ncol = length(fullNms))
colnames(resMat) <- fullNms


        seedSet <- iterNum # ensures no repeats
        set.seed(seedSet)

        # data
        full <- read.csv("/n/home12/gloewinger/sub_samp2500")[,-c(1,2,4,5)] #read.csv( paste0("/n/home12/gloewinger/fscv_subset") )
        colnames(full)[c(1,2)] <- c("Study", "Y_DA")
        
        # select subset
        studs <- sample(unique(full$Study), 
                        K,
                        replace = FALSE)
        full <- full[full$Study %in% studs, ]
        
        full$Study <- as.numeric( as.factor( full$Study  ) ) # reset study labels to be 1:K

        XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
        Yindx <- which(XY == "Y_") 
        
        # sub sample columns
        XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
        Yindx <- which(XY == "Y_") 
        Xindx <- seq(1, ncol(full) )[-c(1, Yindx)] # all cols except outcome and Study
        colSamp <- Xindx[seq(1, length(Xindx), by =  colSubSamp) ]
        full <- full[, c(1, Yindx, colSamp) ]
        
        # find X_indx and Y_indx NEED TO CHANGE IF HAVE MULTIVARIATE OUTCOME
        XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
        Yindx <- which(XY == "Y_") 
        Xindx <- seq(1, ncol(full) )[-c(1, Yindx)] # all cols except outcome and Study
        colnames(full)[Yindx] <- "Y"  # NEED TO REMOVE IF HAVE MULTIVARIATE OUTCOME
        numCovs <- length(Xindx)
        
        # vector of true indicators of whether 0 or not -- these are fake and are just to make the rest of the code align -- all support measures will be meaningless
        z <- rep(0, numCovs)
        trueZ <- matrix(0, nrow = K_train + 1, ncol = numCovs + 1)

        if(dataPro == 2){
            full[,Yindx] <- log(full[,Yindx] + 1e-6) 
            
        }else if(dataPro == 3){
            
            for(j in 1:K){
                
                # log quantile method
                ids <- which(full$Study == j)
                full[ids, Yindx] <- log(full[ids, Yindx] + 
                                        (quantile(full[ids, Yindx])[4] / quantile(full[ids, Yindx])[2])^2 )
                
                rm(ids)
                }
           
            
        }else if(dataPro == 4){
            
            for(j in 1:K){
                
                # Yeo-Johnson Transform
                ids <- which(full$Study == j)
                pt <- car::powerTransform(Y ~., data = full[ids, -1], family = "yjPower") 
                full$Y[ids] <- yjPower(full$Y[ids], pt$lambda)
                rm(ids)
                
            }
            
        
        }
        

        # split multi-study
        mtTest <- multiTaskSplit(data = full, split = 0.5) # make training sets
        full <- mtTest$train
        mtTest <- mtTest$test
        full$Study <- as.numeric( as.factor(full$Study) )

        #################
        # Study Labels
        #################
        full$Study <- as.numeric( as.factor(full$Study) )
        countries <- unique(full$Study) # only include countries that have both
        K <- length(countries) # number of training countries
        
        ###########################
        # sub-sample observations
        ###########################
        n_k <- min(nrow(full) / K, n_k)
        # inclIndx <-  seq(1, nrow(full), length = n_k * K 
        inclIndx <- seq(1, nrow(full), length = n_k * K ) # sample n_k from each electrode at evenly spaced intervals
        inclIndx <- round(inclIndx) # make sure they are integers
        full <- full[inclIndx,]
        ####################################
        # scale covariates
        ####################################
        
        if(scaleInd == TRUE){
            for(kk in 1:K){
                
                # indices
                kkIndx <- which(full$Study == kk) # train set indices
                kkTest <- which(mtTest$Study == kk) # test set indices
                nFull <- length(kkIndx) # sample size of merged
                
                # scale Covaraites
                means <- colMeans( as.matrix( full[kkIndx,]  ) )
                sds <- sqrt( apply( as.matrix( full[kkIndx,]  ), 2, var) *  (nFull - 1) / nFull )  # use mle formula to match with GLMNET
                
                # columns 1 and 2 are Study and Y
                for(column in Xindx ){
                    
                    # center scale
                    full[kkIndx, column] <- (full[kkIndx, column ] - means[column ]) / sds[column ]
                    mtTest[kkTest, column] <- (mtTest[kkTest, column ] - means[column ]) / sds[column ]
                    
                }
            
            }
            
            
        }
        
        if(Yscale){
            
            for(kk in 1:K){
                
                # indices
                kkIndx <- which(full$Study == kk) # train set indices
                kkTest <- which(mtTest$Study == kk) # test set indices
                nFull <- length(kkIndx) # sample size of merged
                
                # calculate statistics
                sds_y <- as.numeric( sqrt( var( as.matrix( full[kkIndx, Yindx]  ) ) *  (nFull - 1) / nFull ) )
                means_y <- as.numeric( mean( full[kkIndx, Yindx]   ) )
            
                # standardize
                full[kkIndx, Yindx] <- (full[kkIndx, Yindx] - means_y) / sds_y
                mtTest[kkTest, Yindx] <- (mtTest[kkTest, Yindx] - means_y) / sds_y

            }
            
        }   
        
        #test <- test[,-1] # remove Study labels from test set
       # mtTest <- mtTest  # remove study labels from multi-task test set
        nFull <- nrow(full)
        rownames(mtTest) <- 1:nrow(mtTest)
        rownames(full) <- 1:nFull
        
        
        rho <- rho[ rho < length(Xindx) ] # remove any rhos that are too large for number of features in dataset
        lambda <- lambda[ lambda >= 0.1 ]
        
        tune.grid_MS2 <- as.data.frame(  expand.grid( 0, lambda, rho) ) # tuning parameters to consider
        tune.grid_MSZ <- as.data.frame(  expand.grid( lambda, 0, lambdaZ, rho) ) # tuning parameters to consider
        tune.grid_MSZ_2 <- as.data.frame(  expand.grid( 0, lambda, lambdaZ, rho) ) # tuning parameters to consider
        tune.grid_MSZ_3 <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
        tune.grid_beta <- as.data.frame(  expand.grid( 0, lambdaBeta, 0, rho) ) # tuning parameters to consider
        
        colnames(tune.grid_MS2) <- c("lambda1", "lambda2", "rho")
        colnames(tune.grid_MSZ) <- colnames(tune.grid_MSZ_2) <- colnames(tune.grid_MSZ_3) <- colnames(tune.grid_beta) <- c("lambda1", "lambda2", "lambda_z","rho")
        
        # order correctly
        tune.grid_MSZ <- tune.grid_MSZ[  order(-tune.grid_MSZ$rho,
                                               tune.grid_MSZ$lambda1,
                                               -tune.grid_MSZ$lambda_z,
                                               decreasing=TRUE),     ]
        
        # order correctly
        tune.grid_MSZ_2 <- tune.grid_MSZ_2[  order(-tune.grid_MSZ_2$rho,
                                                   -tune.grid_MSZ_2$lambda2,
                                                   -tune.grid_MSZ_2$lambda_z,
                                                   decreasing=TRUE),     ]
        
        # order correctly
        tune.grid_MSZ_3 <- tune.grid_MSZ_3[  order(-tune.grid_MSZ_3$rho,
                                                   tune.grid_MSZ_3$lambda1,
                                                   -tune.grid_MSZ_3$lambda2,
                                                   decreasing=TRUE),     ]
        
        tune.grid <- as.data.frame(  expand.grid(
            c(lambda) , # 0 # add 0 but not to glmnet because that will cause problems
            rho)
        ) # tuning parameters to consider
        colnames(tune.grid) <- c("lambda", "rho")
        
        # glmnet for ridge
        # tune.gridGLM <- as.data.frame( expand.grid(seq(0,1,by=0.1), lambda) ) # Ridge
        tune.gridGLM <- as.data.frame( cbind( 0, lambda) ) # Ridge
        
        colnames(tune.gridGLM) <- c("alpha", "lambda")
        
        timeStartTotal <- Sys.time()
 
        #####################################################################
        timeStartTotal <- Sys.time()
        
        for(r in 1:length(rhoVec)){
            
            rCnt <- r
            print(paste0("r: ", r))
            resIndx <- (r - 1) * nmLen # saving in matrix indx
            r <- rhoVec[r] # rho value
            # resIndx2 <- which(colnames(resMat) == paste0("MT_mtLasso_", r, "_pair"))
            
            ##############
            # GLMNET Lasso - MultiTask
            ##############
            # errorVec <- errorVec[ suppVec <= r ] # errors for which rho is smaller than the max
            # sV <- suppVec[ suppVec <= r ]
            # lambs <- lambSeq[ suppVec <= r ]
            # 
            # lambdaStar <- lambs[which.min(errorVec)]
            # 
            # # best lasso with s rho constraint
            # betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = lambdaStar) ) )
            # betaEst <- as.matrix(betaEst)
            # 
            # resMat[iterNum, resIndx + 1] <- multiTaskRmse(data = mtTest, beta = betaEst)
            # resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betaEst, intercept = TRUE)
            # rm(betaEst, lambdaStar, errorVec, suppVec, lambSeq)
            # 
            
            ##############
            # OSE L0Learn
            ##############
            # print(paste("iteration: ", iterNum, " L0Learn OSE"))
            # predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
            # betaMat <- matrix(NA, ncol = K, nrow = length(Xindx) + 1) # matrix of betas from each study
            # res <- vector(length = K) # store auc
            # L0_tune <- matrix(NA, nrow = K, ncol = ncol(tune.grid) ) # save best parameter values
            # L0_tune <- as.data.frame(L0_tune)
            # colnames(L0_tune) <- colnames(tune.grid)
            # 
            # supMat <- matrix(NA, nrow = K, ncol = 4)
            # 
            # for(j in 1:K){
            #     
            #     print(j)
            #     sdY <- 1 # set to 1 for now so we DO NOT adjust as glmnet() #sd(full$Y[indx]) * (n_k - 1) / n_k #MLE
            #     gm <- tune.grid$lambda / (sdY * 2) # convert into comparable numbers for L0Learn
            # 
            #     # fit l0 model on jth study
            #     cvfit = L0Learn.cvfit(x = as.matrix(full[, Xindx]),
            #                           y = as.vector(full[,j]),
            #                           nFolds = nfold, # caused problems for low n_k settings
            #                           seed = 1,
            #                           penalty="L0L2",
            #                           nGamma = length(gm),
            #                           #gammaMin = min(gm), # min and max numbers of our 2 parameter that is comaprable
            #                           #gammaMax = max(gm),
            #                           algorithm = "CD", #"CDPSI",
            #                           maxSuppSize = max(tune.grid$rho)#, # largest that we search
            #                           # scaleDownFactor = 0.99
            #     )
            #     
            #     # optimal tuning parameters
            #     optimalGammaIndex <- which.min( lapply(cvfit$cvMeans, min) ) # index of the optimal gamma identified previously
            #     optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
            #     optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
            #     L0LearnCoef <- coef(cvfit, lambda=optimalLambda, gamma = cvfit$fit$gamma[optimalGammaIndex] )
            # 
            #     # save tuned parameter values
            #     rhoStar <- sum(  as.vector(L0LearnCoef)[-1] != 0   ) # cardinality
            #     L0_tune$lambda[j] <- cvfit$fit$gamma[optimalGammaIndex] * (2 * sdY) # put on scale used by gurobi version below
            #     L0_tune$rho[j] <- rhoStar
            # 
            #     # use L0Learn coefficients as warm starts
            #     betaMat[,j] <- L0LearnCoef <- as.vector(L0LearnCoef) # save coefficients -- use "betas" as warm start for later
            # 
            #     # stack matrix
            #     predsMat[,j] <- cbind( 1, as.matrix( full[,Xindx ] ) ) %*% L0LearnCoef
            # 
            #     rm(cvfit)
            # }
            # 
            # rm(sdY)
            # 
            # zAvg <- I( rowMeans(betaMat)[-1] != 0) * 1 # stacking
            # 
            # resMat[iterNum, 165] <- mean(L0_tune$rho) # cardinality of support
            # resMat[iterNum, 166] <- sum( zAvg ) # cardinality of support
            # resMat[iterNum, 222] <- multiTaskRmse(data = mtTest, beta = betaMat)
            # 
            # rm(betaMat, res, zAvg)
            
            ############
            # OSE L0
            ############
            timeStart1 <- Sys.time()
            b <- matrix(0, ncol = K, nrow = numCovs + 1)
            tune.grid <- as.data.frame(  expand.grid(
                c(lambda) , # 0 # add 0 but not to glmnet because that will cause problems
                rho)
            ) # tuning parameters to consider
            colnames(tune.grid) <- c("lambda", "rho")
            resIndx2 <- which(colnames(resMat) == paste0("MT_oseL0_", rCnt, "_pair"))
            
            tune.grid_OSE <- data.frame(lambda1 = unique(lambda),
                                        lambda2 = 0,
                                        lambda_z = 0,
                                        #rho = numCovs
                                        rho = r)
            
            tune.grid_OSE <- unique( tune.grid_OSE )
            
            L0_tune <- sparseCV_iht_par(data = full,
                                        tune.grid = tune.grid_OSE,
                                        hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                        method = "MS_z3", #"MS_z_fast", # this does not borrow information across the active sets
                                        nfolds = nfold,
                                        cvFolds = 5,
                                        juliaPath = juliaPath,
                                        juliaFnPath = juliaFnPath,
                                        messageInd = TRUE,
                                        LSitr = LSitr,
                                        LSspc = LSspc,
                                        maxIter = maxIter_cv,
                                        threads = tuneThreads,
                                        WSmethod = WSmethod,
                                        ASpass = ASpass
            )
            
            L0_tune <- L0_tune$best # parameters
            ridgeLambda <- L0_tune$lambda1 
            ridgeRho <- L0_tune$rho
            if(!rdgLmd)   ridgeLambda <- 0 # set it to 0 if rdgLmd is false
            
            # rm(L0_MS_z3)
            # L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study
            # 
            # warm start
            warmStart = L0_MS_z3(X = as.matrix( full[ , Xindx ]) ,
                                 y = as.matrix( full[, Yindx] ),
                                 rho = min(L0_tune$rho * 4, numCovs -1),
                                 study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                 beta = b,
                                 lambda1 = L0_tune$lambda1,
                                 lambda2 = 0,
                                 lambda_z = 0,
                                 scale = TRUE,
                                 maxIter = maxIter_train,
                                 localIter = 0,
                                 WSmethod = WSmethod,
                                 ASpass = ASpass
            )
            
            # final model
            betas = L0_MS_z3(X = as.matrix( full[ , Xindx ]) ,
                             y = as.matrix( full[, Yindx] ),
                             rho = L0_tune$rho,
                             study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                             beta = warmStart,
                             lambda1 = L0_tune$lambda1,
                             lambda2 = 0,
                             lambda_z = 0,
                             scale = TRUE,
                             maxIter = maxIter_train,
                             localIter = localIters,
                             WSmethod = WSmethod,
                             ASpass = ASpass
            )
            
            timeEnd1 <- Sys.time()
            
            # resMat[iterNum, 233] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
            # 
            # resMat[iterNum, 167] <- L0_tune$rho # cardinality of support
            resMat[iterNum, resIndx + 2] <- multiTaskRmse(data = mtTest, beta = betas)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betas, intercept = TRUE)
            
            rm(L0_tune, betas)
            
            
            ####################################
            # common support L0 regularization with ||beta - betaBar|| penalty
            # glPenalty = TRUE, ip = TRUE
            ####################################
            # random initialization for betas
            #betas <- matrix( 0, nrow = numCovs + 1, ncol = K )#matrix( rnorm( K * p ), ncol = K )
            
            print(paste("iteration: ", iterNum, " Group IP"))
            glPenalty <- 1
            ip <- TRUE
            resIndx2 <- which(colnames(resMat) == paste0("MT_msP1_L0_", rCnt, "_pair"))
            
            # tune multi-study with l0 penalty with GL Penalty = TRUE
            tune.grid_MS2$rho <- r
            tune.grid_MS2 <- unique(tune.grid_MS2)
            
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid_MS2,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS2", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = TRUE,
                                       LSitr = LSitr,
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       maxIter = maxIter_cv
            )
            
            MSparams <- tuneMS$best # parameters
            
            # rm(L0_MS2)
            # L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
            # 
            # warm start for 4 * rho and no betaBar regularization
            warmStart = L0_MS2(X = as.matrix( full[ , Xindx ] ),
                               y = as.matrix( full[, Yindx] ),
                               rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                               study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                               beta = b,
                               lambda1 = MSparams$lambda1,
                               lambda2 = 0, # use 0 as warm start
                               scale = TRUE,
                               maxIter = maxIter_train,
                               localIter = 0
            )
            
            # final model
            betasMS = L0_MS2(X = as.matrix( full[ , Xindx ]) ,
                             y = as.matrix( full[, Yindx] ),
                             rho = MSparams$rho,
                             study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                             beta = warmStart,
                             lambda1 = MSparams$lambda1,
                             lambda2 = MSparams$lambda2,
                             scale = TRUE,
                             maxIter = maxIter_train,
                             localIter = localIters
            )
            
            # resMat[iterNum, 96] <- MSparams$lambda2 # tuning parameter
            # 
            # resMat[iterNum, 169] <- MSparams$rho # cardinality of support
            # 
            resMat[iterNum, resIndx + 3] <- multiTaskRmse(data = mtTest, beta = betasMS)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betasMS, intercept = TRUE)
            
            rm(MSparams, tuneMS, betasMS, warmStart)
            
            ########################################################
            #  L0 regularization with ||z - zbar|| penalty and frobenius norm
            # glPenalty = 2
            ####################################
            timeStart1 <- Sys.time()
            resIndx2 <- which(colnames(resMat) == paste0("MT_msP2_L0_", rCnt, "_pair"))
            
            predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
            res <- resS <- vector(length = K) # store support prediction
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZ, r) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                       tune.grid_MSZ_5$lambda1,
                                                       -tune.grid_MSZ_5$lambda_z,
                                                       decreasing=TRUE),     ]
            
            tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
            
            lambdaZScale <- rhoScale(K = K, 
                                     p = numCovs, 
                                     rhoVec = tune.grid_MSZ_5$rho, 
                                     itrs = 100000,
                                     seed = 1)
            
            tune.grid_MSZ_5 <- tuneZscale(tune.grid = tune.grid_MSZ_5, 
                                          rhoScale = lambdaZScale)
            
            # tune z - zbar and rho
            # tune multi-study with l0 penalty with z - zbar and beta - betaBar penalties
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid_MSZ_5,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            
            rm(L0_MS_z)
            L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
            
            # warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
            warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                                y = as.matrix( full[, Yindx] ),
                                rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                                study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                beta = b,
                                lambda1 = max(MSparams$lambda1, 1e-6), # ensure theres some regularization given higher rho for WS
                                lambda2 = 0, #MSparams$lambda2,
                                lambda_z = 0,
                                scale = TRUE,
                                maxIter = maxIter_train,
                                localIter = 0,
                                WSmethod = WSmethod,
                                ASpass = ASpass
            )
            
            # final model
            betasMS = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                              y = as.matrix( full[, Yindx] ),
                              rho = MSparams$rho,
                              study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                              beta = warmStart,
                              lambda1 = MSparams$lambda1,
                              lambda2 = MSparams$lambda2,
                              lambda_z = MSparams$lambda_z,
                              scale = TRUE,
                              maxIter = maxIter_train,
                              localIter = localIters,
                              WSmethod = WSmethod,
                              ASpass = ASpass
            )
            
            
            
            # print(difftime(timeEnd, timeStart, units='mins'))
            # resMat[iterNum, 231] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
            
            resMat[iterNum, resIndx + 4] <- multiTaskRmse(data = mtTest, beta = betasMS)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betasMS, intercept = TRUE)
            
            rm(MSparams, tuneMS, res, resS, betasMS, warmStart )
            ########################################################
            ####################################
            # MS5 -- different support with beta - betaBar penalty AND z- zbar penalty, no frobenius
            ####################################
            # share info on the beta - betaBar AND ||z - zbar|| 
            timeStart1 <- Sys.time()
            resIndx2 <- which(colnames(resMat) == paste0("MT_msP5_L0_", rCnt, "_pair"))
            
            glPenalty <- 4
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, lambdaBeta, lambdaZ, r) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                       -tune.grid_MSZ_5$lambda2,
                                                       -tune.grid_MSZ_5$lambda_z,
                                                       decreasing=TRUE),     ]
            
            tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
            
            lambdaZScale <- rhoScale(K = K, 
                                     p = numCovs, 
                                     rhoVec = tune.grid_MSZ_5$rho, 
                                     itrs = 100000,
                                     seed = 1)
            
            tune.grid_MSZ_5 <- tuneZscale(tune.grid = tune.grid_MSZ_5, 
                                          rhoScale = lambdaZScale)
            # tune z - zbar and rho
            # tune multi-study with l0 penalty with z - zbar and beta - betaBar penalties
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid_MSZ_5,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       maxIter = maxIter_cv,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            ###################################################
            rm(L0_MS_z)
            L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
            
            # warm start
            warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                                y = as.matrix( full[, Yindx] ),
                                rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                                study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                beta = b,
                                lambda1 = max(MSparams$lambda1, 1e-6), # ensure theres some regularization given higher rho for WS
                                lambda2 = 0, #MSparams$lambda2,
                                lambda_z = 0, #MSparams$lambda_z,
                                scale = TRUE,
                                maxIter = maxIter_train,
                                localIter = 0,
                                WSmethod = WSmethod,
                                ASpass = ASpass
            )
            
            # final model
            betasMS = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                              y = as.matrix( full[, Yindx] ),
                              rho = MSparams$rho,
                              study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                              beta = warmStart,
                              lambda1 = MSparams$lambda1,
                              lambda2 = MSparams$lambda2,
                              lambda_z = MSparams$lambda_z,
                              scale = TRUE,
                              maxIter = maxIter_train,
                              localIter = localIters ,
                              WSmethod = WSmethod,
                              ASpass = ASpass
            )
            
            timeEnd1 <- Sys.time()
            
            # resMat[iterNum, 232] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
            # 
            # resMat[iterNum, 199] <- MSparams$lambda2 # tuning parameter
            # resMat[iterNum, 200] <- MSparams$lambda_z # tuning parameter
            # resMat[iterNum, 201] <- MSparams$rho # cardinality of support
            resMat[iterNum, resIndx + 6] <- multiTaskRmse(data = mtTest, beta = betasMS)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betasMS, intercept = TRUE)
            
            rm(MSparams, tuneMS, betasMS, warmStart)
            
            ########################################################
            # ******************************************************
            ########################################################
            
            ####################################
            # MS4 -- different support with beta - betaBar penalty AND NO frobenius norm AND NO z- zbar penalty
            ####################################
            # share info on the beta - betaBar but no ||z - zbar|| penalty (currently but could do it if changed tuning grid)
            glPenalty <- 4
            resIndx2 <- which(colnames(resMat) == paste0("MT_msP4_L0_", rCnt, "_pair"))
            
            tune.grid_beta <- as.data.frame(  expand.grid( 0, lambdaBeta, 0, r) ) # tuning parameters to consider
            colnames(tune.grid_beta) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_beta <- tune.grid_beta[  order(-tune.grid_beta$rho,
                                                     -tune.grid_beta$lambda2,
                                                     decreasing=TRUE),     ]
            
            tune.grid_beta <- unique(tune.grid_beta)
            
            # tune multi-study with l0 penalty with only beta - betaBar penalties
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid_beta,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       maxIter = maxIter_cv,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # tuned parameters
            
            rm(L0_MS_z)
            L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
            
            # warm start
            warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                                y = as.matrix( full[, Yindx] ),
                                rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                                study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                beta = b,
                                lambda1 = max(MSparams$lambda1, 1e-5),
                                lambda2 = 0, #MSparams$lambda2,
                                lambda_z = 0, #MSparams$lambda_z,
                                scale = TRUE,
                                maxIter = maxIter_train,
                                localIter = 0,
                                WSmethod = WSmethod,
                                ASpass = ASpass
            )
            
            # final model
            betasMS = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                              y = as.matrix( full[, Yindx] ),
                              rho = MSparams$rho,
                              study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                              beta = warmStart,
                              lambda1 = MSparams$lambda1,
                              lambda2 = MSparams$lambda2,
                              lambda_z = MSparams$lambda_z,
                              scale = TRUE,
                              maxIter = maxIter_train,
                              localIter = localIters,
                              WSmethod = WSmethod,
                              ASpass = ASpass
            )
            
            # resMat[iterNum, 99] <- MSparams$lambda2 # tuning parameter
            # resMat[iterNum, 175] <- MSparams$rho # cardinality of support
            # 
            resMat[iterNum, resIndx + 7] <- multiTaskRmse(data = mtTest, beta = betasMS)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betasMS, intercept = TRUE)
            
            rm(MSparams, tuneMS, betasMS, warmStart)
            ########################################################
            
            ####################################################
            # Group Penalty = 3 (i.e., just frobenius norm)
            ####################################################
            print(paste("iteration: ", iterNum, " GPenalty = 3"))
            ####################################
            # common support L0 regularization with just Frobenius Norm (no other penalty): glPenalty = 3, IP = TRUE
            ####################################
            # MS3
            glPenalty <- 3
            ip = TRUE
            tune.grid$rho <- r
            tune.grid <- unique(tune.grid)
            resIndx2 <- which(colnames(resMat) == paste0("MT_msP3_L0_", rCnt, "_pair"))
            
            # tune multi-study with l0 penalty with GL Penalty = TRUE
            
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = TRUE,
                                       LSitr = LSitr,
                                       LSspc = LSspc,
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads
            )
            
            MSparams <- tuneMS$best # parameters
            
            rm(L0_MS)
            L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
            
            # warm start
            warmStart = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                              y = as.matrix( full[, Yindx] ),
                              rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                              study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                              beta = b,
                              lambda = MSparams$lambda,
                              scale = TRUE,
                              maxIter = maxIter_train,
                              localIter = 0
            )
            
            # final model
            betasMS = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                            y = as.matrix( full[, Yindx] ),
                            rho = MSparams$rho,
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = warmStart,
                            lambda = MSparams$lambda,
                            scale = TRUE,
                            maxIter = maxIter_train,
                            localIter = localIters
            )
            
            # resMat[iterNum, 100] <- MSparams$lambda # tuning parameter
            # resMat[iterNum, 177] <- MSparams$rho # cardinality of support
            # 
            resMat[iterNum, resIndx + 5] <- multiTaskRmse(data = mtTest, beta = betasMS)
            resMat[iterNum, (resIndx2):(resIndx2+1)] <- suppHet(betasMS, intercept = TRUE)
            
            rm(MSparams, tuneMS, betasMS, warmStart)
            
        }
        
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        
        ####################################################
        # convex version of MS (no cardinality constraints)
        ####################################################
        ####################################
        # ||beta - betaBar|| and NO Frobenius norm no cardinality constraints
        # Convex, IP = FALSE
        ####################################
        
        tune.grid_MS2$rho <- length(Xindx) # full support
        tune.grid_MS2 <- unique(tune.grid_MS2) # convex -- no IP selection
        
        # tune multi-study with l0 penalty with GL Penalty = TRUE
        
        tuneMS <- sparseCV_iht_par(data = full,
                                   tune.grid = tune.grid_MS2,
                                   hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                   method = "MS2", # could be L0 for sparse regression or MS # for multi study
                                   nfolds = nfold,
                                   cvFolds = 5,
                                   juliaPath = juliaPath,
                                   juliaFnPath = juliaFnPath,
                                   messageInd = FALSE,
                                   LSitr = NA,
                                   LSspc = NA, # convex
                                   maxIter = maxIter_cv,
                                   threads = tuneThreads 
        )
        
        MSparams <- tuneMS$best # parameters
        
        rm(L0_MS2)
        L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
        
        betasMS = L0_MS2(X = as.matrix( full[ , Xindx ]) ,
                         y = as.matrix( full[, Yindx] ),
                         rho = MSparams$rho,
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = b,
                         lambda1 = MSparams$lambda1,
                         lambda2 = MSparams$lambda2,
                         scale = TRUE,
                         maxIter = maxIter_train,
                         localIter = 0 # convex
        )
        
        nmID <- which(fullNms == "msP1_con")
        
        # resMat[iterNum, 98] <- MSparams$lambda2 # tuning parameter
        # resMat[iterNum, 173] <- MSparams$rho # cardinality of support
        # 
        resMat[iterNum, nmID] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS)
        ########################################################
        
        ####################################
        # Convex MS, glPenalty = 3, IP FALSE: Just Frobenius norm and no other sharing of information
        ####################################
        glPenalty <- 3
        ip <- FALSE
        tune.grid2 <- tune.grid
        tune.grid2$rho <- length(Xindx) # full support
        tune.grid2 <- unique(tune.grid2)
        nmID <- which(fullNms == "msP3_con")
        
        tuneMS <- sparseCV_iht_par(data = full,
                                   tune.grid = tune.grid2,
                                   hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                   method = "MS", # could be L0 for sparse regression or MS # for multi study
                                   nfolds = nfold,
                                   cvFolds = 5,
                                   juliaPath = juliaPath,
                                   juliaFnPath = juliaFnPath,
                                   messageInd = TRUE,
                                   LSitr = NA, # convex
                                   LSspc = NA,
                                   maxIter = maxIter_cv,
                                   threads = tuneThreads
        )
        
        MSparams <- tuneMS$best # parameters
        
        rm(L0_MS)
        L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
        
        betasMS = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                        y = as.matrix( full[, Yindx] ),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = b,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = maxIter_train,
                        localIter = 0 # convex
        )
        
        # resMat[iterNum, 101] <- MSparams$lambda # tuning parameter
        # resMat[iterNum, 179] <- MSparams$rho # cardinality of support
        #resMat[iterNum, 180] <- sum( zStack ) # cardinality of support
        #resMat[iterNum, 216] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, nmID ] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS)
        ########################################################
        #####################################################################
        ########################################################
        # ***multi-task learning***
        ########################################################
        
        package <- "RMTL"
        if (!require(package, character.only=T, quietly=T)) {
            install.packages(package, 
                             repos='http://cran.us.r-project.org',
                             dependencies = TRUE
            )
            library(package, character.only=T)
        }
        ##############
        # RMTL - L21
        ##############
        # format data for multi-task learning
        Xlist <- list(length = K)
        Ylist <- list(length = K)
        
        for(kk in 1:K){
            idx <- which(full$Study == kk)
            Xlist[[kk]] <- cbind(1, full[idx, Xindx]) # same design matrix for all
            Ylist[[kk]] <- full$Y[idx]
            
            full <- full[-idx,]
        }
        
        ##############
        # RMTL - Trace
        ##############
        # format data for multi-task learning
        trcIndx <- which(colnames(resMat) == "MTL_trace")
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="Trace", 
                        Lam1_seq = lambda,
                        Lam2 = 0,
                        nfolds = nfold) #, 
        #parallel=TRUE, 
        #ncores = 3)
        
        model <- MTL(X = Xlist, 
                     Y = Ylist, 
                     type="Regression", 
                     Regularization="Trace",
                     Lam1=cvfitc$Lam1.min, 
                     Lam2 = 0, #cvfitc$Lam2.min, 
                     opts=list(init=0,  tol=10^-6,
                               maxIter=1500), 
                     Lam1_seq=cvfitc$Lam1_seq
        )
        
        resMat[iterNum, trcIndx - 1] <-  sum( model$W[-1,] != 0 ) / K # size of xupport
        resMat[iterNum, trcIndx] <- multiTaskRmse(data = mtTest, beta = model$W)
        
        ##############
        # RMTL - L2L1
        ##############
        
        l2l1indx <- which(colnames(resMat) == "MT_lasso_low")
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="L21", 
                        nfolds = nfold, 
                        Lam1_seq = lambda,
                        Lam2 = 0) #, #lambda,
        
        model <- MTL(X = Xlist, 
                     Y = Ylist, 
                     type="Regression", 
                     Regularization="L21",
                     Lam1=cvfitc$Lam1.min, 
                     Lam2 = 0, 
                     opts=list(init=0,  tol=10^-6,
                               maxIter=1500), 
                     Lam1_seq=cvfitc$Lam1_seq
        )
        
        resMat[iterNum, l2l1indx + 1] <-  S <- sum( model$W[-1,] != 0 ) / K # size of support
        resMat[iterNum, l2l1indx] <- multiTaskRmse(data = mtTest, beta = model$W)
        
        rm(model)
        
        #################################
        # L2L1
        #################################
        lambda_vector <- sort( unique( c(0.0001, 0.001, 0.01, 5,10, 50, 
                                         exp(-seq(0,5, length = 50))) ), 
                               decreasing = TRUE ) 
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="L21", 
                        nfolds = nfold, 
                        Lam1_seq = lambda_vector, 
                        Lam2 = 0) 
        
        lambSeq <- cvfitc$Lam1_seq
        len <- length(lambSeq)
        errorVec <- cvfitc$cvm
        errorVector <- cvfitc$cvm
        
        suppVec <- vector(length = len )
        
        for(lam in 1:len){
            
            model <- MTL(X = Xlist, 
                         Y = Ylist, 
                         type="Regression", 
                         Regularization="L21",
                         Lam1=lambSeq[lam], 
                         Lam2 = 0, #cvfitc$Lam2.min, 10^seq(1,-4, -1)
                         opts=list(init=0,  
                                   tol=10^-6,
                                   maxIter=1500), 
                         Lam1_seq=cvfitc$Lam1_seq
            )
            
            suppVec[lam] <- sum( model$W[-1,] != 0 ) / K
            
        }
        
        rm(model)
        
        for(r in length(rhoVec):1 ){
            
            rCnt <- r
            resIndx <- (r - 1) * nmLen # saving in matrix indx
            r <- rhoVec[r] # rho value
            ##########################################
            # only run if solution isn't sparse enough
            #  if( S > r ){
            errorVec <- errorVector
            errorVec <- errorVec[ suppVec <= r ] # errors for which rho is smaller than the max
            sV <- suppVec[ suppVec <= r ]
            lambs <- lambSeq[ suppVec <= r ]
            
            lambStar <- lambs[which.min(errorVec)]
            
            # best model with support
            model <- MTL(X = Xlist, 
                         Y = Ylist, 
                         type="Regression", 
                         Regularization="L21",
                         Lam1=lambStar, 
                         Lam2 = 0, #cvfitc$Lam2.min, 10^seq(1,-4, -1)
                         opts=list(init=0,  tol=10^-6,
                                   maxIter=1500), 
                         Lam1_seq = lambSeq
            )
            
            # FIX INDICES "mtl_L2L1_supp"
            
            suppSize <- sum( model$W[-1,] != 0 ) / K 
            
            if(suppSize <= r){
                nmIndx <- which(colnames(resMat) == paste0("MT_mtLasso_", rCnt, "_pair") )
                resMat[iterNum, resIndx + 1] <- multiTaskRmse(data = mtTest, beta = model$W)
                resMat[iterNum, nmIndx:(nmIndx + 1)] <- suppHet( as.matrix(model$W), intercept = TRUE)
                
            }
            
            rm(model)

        }
        
        rm(Xlist, Ylist)
        ########################################################
        # print(paste("iteration: ", iterNum, " Complete!"))
        # print(resMat[iterNum,])
        # 
        # # time difference
        # timeEnd <- Sys.time()
        # print(difftime(timeEnd, timeStart, units='mins'))
        # resMat[iterNum, 140] <- as.numeric(difftime(timeEnd, timeStart, units='mins'))
        # 
        # timeEndTotal <- Sys.time()
        # resMat[iterNum, 234] <- as.numeric(difftime(timeEndTotal, timeStartTotal, units='mins'))
        ########################
        # save results
        ########################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        
        #####################################################################
        