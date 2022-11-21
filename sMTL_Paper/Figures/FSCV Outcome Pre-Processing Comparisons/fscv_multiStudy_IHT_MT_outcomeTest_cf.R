# library(pROC)
# test which version of the outcome works best
library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)
library(car)

source("rmse_mt.R") 
source("sparseFn_iht_test_MT.R") 

#sims6 <- read.csv("sparseParam")
cluserInd <- TRUE # whether running on computer or on cluster

save.folder <- "/n/home12/gloewinger/tung_MT_outcome"
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
    runNum <- 10
    setwd("~/Desktop/Research")
    iterNum <- 1
    n_k <- 1000
    colSubSamp <- 25 # sample every colSubSamp^th row (so = 1 means use all of them and 2 samples every other)
    
    # Julia paths
    juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
    # juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/"
    juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune/"
    
}

Sys.setenv(JULIA_BINDIR = juliaPath)

# simulation parameters
K <- K_train <- 4

#######################
Yscale <- TRUE # scale outcome

totalSims <- 30
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
dataPro <- 4
Yscale <- FALSE

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

nfoldL0_ose <- min( 5, K) # 5 fold maximum

nfoldL0_ose <- nfold <- 3 # to speed things up

rho <- seq(5, 25, by = 5) #c(seq(2, 10, by = 2), seq(15, 20, by = 5))  # this works: 3/19/22 , just changed for sparse c(seq(2, 10, by = 2), seq(15, 50, by = 5), seq(60, 90, by = 10)) 


rho <- rho[rho <= round(2500 / colSubSamp) ]

rhoDiff <- round( mean( diff(rho) ) / 2 ) # average interval

lambda <- sort( unique( c(1e-6, 1e-5, 0.0001, 0.001, 0.01, 5,10, 50, 100, 200, 
                          exp(-seq(0,5, length = 15))
) ), decreasing = TRUE ) 

lambdaShort <- sort( unique( c(
    exp(-seq(0,5, length = 5)),
    5,10, 50, 100, 250, 500, 1000, 2500, 5000, 10000) ),
    
    decreasing = TRUE ) # 2:100

lambda <- lambda[lambda >= 0.01]
lambdaShort <- lambdaShort[lambdaShort >= 0.01]

lambdaZ <- sort( unique( c(0, c( 10^c(-(6:3)), 5 * 10^c(-(6:3))),
                           exp(-seq(0,5, length = 8)),
                           1:3) ),
                 decreasing = TRUE ) # 2:100

lambdaZ <- lambdaZ * min(rho)

lambdaBeta <- c( 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 )

# file name
fileNm <- paste0("tungFSCV_MT_outTest", 
                 "_datPro_",  dataPro, 
                 "_sclX_", scaleInd,
                 "_rhoLen_", length(rho),
                 "_rdgLmd_", rdgLmd,
                 "_MSTn_", MSTn,
                 "_Yscl_", Yscale,
                 "_rhoMax_", max(rho),
                 "_nFld_", nfold,
                 "_LSitr_", LSitr,
                 "_LSspc_", LSspc,
                 "_fitLocal_", localIters,
                 "_Zlen_", length(lambdaZ),
                 "_TnIn_", tuneInd,
                 "_colSub_", colSubSamp,
                 "_n_k_", n_k, 
                 "_out_", outcome)

print(fileNm)

timeStart <- Sys.time()
#######################
scaleInd <- TRUE

# model tuning parameters
nfold <- min( 5, K) # 5 fold maximum
nfold <- 3 # makee equal to 3 for now for speed

L0TuneInd <- TRUE # whether to retune lambda and rho with gurobi OSE (if FALSE then use L0Learn parameters)
L0MrgTuneInd <- TRUE # whether to retune lambda and rho with gurobi Mrg (if FALSE then use L0Learn parameters)
L0_sseTn <- "sse" # tuning for L0 OSE (with gurobi not L0Learn)
MSTn <- "multiTask" #"balancedCV" # tuning for MS (could be "hoso")

lamb <- 0.5

print(fileNm)

lambda <- sort( unique( c(0.0001, 0.001, 0.01, 5,10, 50, 100, 200, 
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

tune.grid_MS2 <- as.data.frame(  expand.grid( lambdaShort, lambdaShort, rho) ) # tuning parameters to consider
tune.grid_MSZ <- as.data.frame(  expand.grid( lambda, 0, lambdaZ, rho) ) # tuning parameters to consider

colnames(tune.grid_MS2) <- c("lambda1", "lambda2", "rho")
colnames(tune.grid_MSZ) <- c("lambda1", "lambda2", "lambda_z","rho")

# order correctly
tune.grid_MSZ <- tune.grid_MSZ[  order(-tune.grid_MSZ$rho, 
                                       tune.grid_MSZ$lambda1,
                                       -tune.grid_MSZ$lambda_z,
                                       decreasing=TRUE),     ]

tune.grid <- as.data.frame(  expand.grid( 
    c(lambda) , # 0 # add 0 but not to glmnet because that will cause problems
    rho)
) # tuning parameters to consider
colnames(tune.grid) <- c("lambda", "rho")

# glmnet for ridge
tune.gridGLM <- as.data.frame( cbind(0, lambda) ) # Ridge
colnames(tune.gridGLM) <- c("alpha", "lambda")

timeStart <- Sys.time()

#####################################################################
print(paste0("start: ", iterNum))

Sys.setenv(JULIA_BINDIR = juliaPath)
L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study

        resMat <- matrix(nrow = totalSims, ncol = 12)
        colnames(resMat) <- c( paste0("rmse", 1:4),
                               paste0("rho", 1:4),
                               paste0("rmse_rdg", 1:4)
                            )
    
        seedSet <- iterNum # ensures no repeats
        set.seed(seedSet)
        
        # data
        full <- read.csv("/n/home12/gloewinger/sub_samp2500")[,-c(1,2,4,5)] #read.csv( paste0("/n/home12/gloewinger/fscv_subset") )
        colnames(full)[c(1,2)] <- c("Study", "Y_DA")
        
        # select subset
        studs <- sample(unique(full$Study), 
                        K,
                        replace = FALSE
                        )
        
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

        K <- length(unique(full$Study))  # number of training countries
        
        # vector of true indicators of whether 0 or not -- these are fake and are just to make the rest of the code align -- all support measures will be meaningless
        z <- rep(0, numCovs)
        trueZ <- matrix(0, nrow = K_train + 1, ncol = numCovs + 1)

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
        Y_orig <- full$Y
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
        
        ## tuning grid
        b <- matrix(0, ncol = K, nrow = numCovs + 1)
        
        tune.grid_OSE <- expand.grid(lambda1 = unique(lambda),
                                     lambda2 = 0,
                                     lambda_z = 0,
                                     #rho = numCovs)
                                     rho = tune.grid$rho)
        # 
        tune.grid_OSE <- unique( tune.grid_OSE )
        
        # ridge
        tune.grid_OSE2 <- tune.grid_OSE
        tune.grid_OSE2$rho <- numCovs
        tune.grid_OSE2 <- unique(tune.grid_OSE2)
        
        timeStartTotal <- Sys.time()

        for(l in 1:4){
            
            dataPro = c("raw", "log", "log_quant", "yj")[l]
            lambdaVec <- vector(length = K) # store yeo-johnson lambdas
            
            #####################
            # process data
            #####################
            # replace with original Y

            full[,Yindx] <- Y_orig
            colnames(full)[Yindx] <- "Y"
            
            if(l == 2){
                full[,Yindx] <- log(full[,Yindx] + 1e-6) 
                
            }else if(l ==  3){
                
                for(j in 1:K){
                    
                    # log quantile method
                    ids <- which(full$Study == j)
                    full[ids, Yindx] <- log(full[ids, Yindx] + 
                                                (quantile(full[ids, Yindx])[4] / quantile(full[ids, Yindx])[2])^2 )
                    
                    rm(ids)
                }
                
                
            }else if(l == 4){
                
                for(j in 1:K){
                    
                    # Yeo-Johnson Transform
                    ids <- which(full$Study == j)
                    pt <- car::powerTransform(Y ~., data = full[ids, -1], family = "yjPower") 
                    full$Y[ids] <- yjPower(full$Y[ids], pt$lambda)
                    lambdaVec[j] <- pt$lambda # store for inverse transform
                    rm(ids)
                    
                }
                
                
            }
            
            # SD = 1
            SD = sd(full$Y)
            full$Y <- full$Y / SD
            
            ############
            # OSE L0
            ############
            timeStart1 <- Sys.time()
            
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
                                        threads = tuneThreads,
                                        WSmethod = WSmethod,
                                        ASpass = ASpass
            )
            
            L0_tune <- L0_tune$best # parameters
            ridgeRho <- L0_tune$rho # save for later for z - zBar penalties
            ridgeLambda <- L0_tune$lambda1 # save for later for z - zBar penalties
            
            # initialize algorithm warm start
            p <- ncol(full) - 1
            b <-  matrix(0, ncol = K, nrow = p ) #rep(0, p) #rnorm( p )
            
            # warm start
            warmStart = L0_MS_z3(X = as.matrix( full[ , -c(1,2) ]) ,
                                 y = as.vector(full$Y),
                                 rho = L0_tune$rho * 3,
                                 study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                 beta = b,
                                 lambda1 = L0_tune$lambda1,
                                 lambda2 = 0,
                                 lambda_z = 0,
                                 scale = TRUE,
                                 maxIter = 10000,
                                 localIter = 0,
                                 WSmethod = WSmethod,
                                 ASpass = ASpass
            )
            
            # final model
            betas = L0_MS_z3(X = as.matrix( full[ , -c(1,2) ]) ,
                             y = as.vector(full$Y),
                             rho = L0_tune$rho,
                             study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                             beta = warmStart,
                             lambda1 = L0_tune$lambda1,
                             lambda2 = 0,
                             lambda_z = 0,
                             scale = TRUE,
                             maxIter = 10000,
                             localIter = 50,
                             WSmethod = WSmethod,
                             ASpass = ASpass
            )
            
            timeEnd1 <- Sys.time()
            
            resMat[iterNum, 4 + l] <- L0_tune$rho # cardinality of support
            
            resMat[iterNum, l] <- multiTaskRmse_MT_transform(data = full, 
                                                                 fn = dataPro,
                                                                 Y_orig = Y_orig, # original y vector
                                                                 lambda = lambdaVec, # lambda from box cox transform
                                                                 beta = betas, 
                                                             SD =SD
                                                             )
            
            rm(L0_tune, warmStart, betas)
            
            ##########################################
            ############
            # ridge
            ############
            timeStart1 <- Sys.time()
            
            L0_tune <- sparseCV_iht_par(data = full,
                                        tune.grid = tune.grid_OSE2,
                                        hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                        method = "MS_z3", #"MS_z_fast", # this does not borrow information across the active sets
                                        nfolds = nfold,
                                        cvFolds = 5,
                                        juliaPath = juliaPath,
                                        juliaFnPath = juliaFnPath,
                                        messageInd = TRUE,
                                        LSitr = LSitr,
                                        LSspc = LSspc,
                                        threads = tuneThreads,
                                        WSmethod = WSmethod,
                                        ASpass = ASpass
            )
            
            L0_tune <- L0_tune$best # parameters
            ridgeRho <- L0_tune$rho # save for later for z - zBar penalties
            ridgeLambda <- L0_tune$lambda1 # save for later for z - zBar penalties
            
            # final model
            betas = L0_MS_z3(X = as.matrix( full[ , -c(1,2) ]) ,
                             y = as.vector(full$Y),
                             rho = L0_tune$rho,
                             study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                             beta = b,
                             lambda1 = L0_tune$lambda1,
                             lambda2 = 0,
                             lambda_z = 0,
                             scale = TRUE,
                             maxIter = 10000,
                             localIter = 0,
                             WSmethod = WSmethod,
                             ASpass = ASpass
                            )
            

            resMat[iterNum, 8 + l] <- multiTaskRmse_MT_transform(data = full, 
                                                                 fn = dataPro,
                                                                 Y_orig = Y_orig, # original y vector
                                                                 lambda = lambdaVec, # lambda from box cox transform
                                                                 beta = betas,
                                                                 SD = SD
                                                                )
            rm(L0_tune, warmStart, betas)
            
            
        }

        #####################################################################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        #####################################################################
        