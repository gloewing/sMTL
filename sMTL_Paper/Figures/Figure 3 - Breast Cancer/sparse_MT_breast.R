# Updates: Feb 26, 2022
# uses active set versions appropriate for each method
library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)
library(stringr)

cluserInd <- TRUE # whether running on computer or on cluster

save.folder <- "/n/home12/gloewinger/breastCancer4"
load.folder <- "~/Desktop/Research"

if(cluserInd){
    
    # if on cluster
    args <- commandArgs(TRUE)
    K <- as.integer( as.numeric(args[1]) ) # number of tasks
    ds <- as.integer( as.numeric(args[2]) ) # dataset index
    studies <- as.integer( as.numeric(args[3]) ) # dataset index
    iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id
    # 
    if(ds == 1){
        full <- read.csv("breastCancer_data")
    }else if(ds == 2){
        full <- read.csv("breastCancer_data_diff") # if ds == 2
        
    }
    
    # remove cols not using
    XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
    Yindx <- which(XY == "Y_") 
    Yrm <- Yindx[-seq(1,K)] 
    Yindx <- Yindx[1:K]
    if( length(Yrm) > 0)   full <- full[,-Yrm]  # only remove if K < length(Yindx) (i.e., fewer than total number)
    Xindx <- seq(1, ncol(full) )[-c(1, Yindx)] # all cols except outcome and Study
    
    colnames(full)[Xindx] <- paste0( "x_", colnames(full)[Xindx] ) 

    # Julia paths
    juliaPath <- "/n/sw/eb/apps/centos7/Julia/1.5.3-linux-x86_64/bin"
    juliaFnPath_MT <- juliaFnPath <- "/n/home12/gloewinger/"
    
}else{
 
    studies <- 17
    K <- 5
    runNum <- 1
    iterNum <- 17
    ds <- 1
    setwd("~/Desktop/Research")
    
    if(ds == 1){
        full <- read.csv("~/Desktop/Research Final/Sparse Multi-Study/GP Data/final/breastCancer_data")
    }else if(ds == 2){
        full <- read.csv("~/Desktop/Research Final/Sparse Multi-Study/GP Data/final/breastCancer_data_diff") # if ds == 2
        
    }

    # remove cols not using
    XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
    Yindx <- which(XY == "Y_") 
    Yrm <- Yindx[-seq(1,K)] 
    Yindx <- Yindx[1:K]
    
    if( length(Yrm) > 0)   full <- full[,-Yrm]

    Xindx <- seq(1, ncol(full) )[-c(1, Yindx)] # all cols except outcome and Study
    colnames(full)[Xindx] <- paste0( "x_", colnames(full)[Xindx] ) 

    # Julia paths
    juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
    juliaFnPath_MT <- juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune MT/"
}

Sys.setenv(JULIA_BINDIR = juliaPath)

set.seed(iterNum)

totalStudies <- length(unique(full$Study)) # total number of studies
testStudy <- iterNum %% totalStudies # find index of test study in cases where it may be that iterNum > totalStudies

testStudy <- ifelse(testStudy == 0, 18, testStudy) # if test study exactly divisible by 18, then %% will be 0 and needs to be set to 18

studies <- min(studies, totalStudies - 1) # make sure you do not select too many
trainStudies <- sample( seq(1, 18)[-testStudy], 
                       studies, 
                       replace = FALSE) # randomly select a sequence of training studies
trainStudies <- sort(trainStudies)

totStudies <- c(trainStudies, testStudy) # all studies to save including test study
Yscale <- TRUE # scale outcome

totalSims <- 90
numCovs <- ncol(full) - K
scaleInd <- TRUE

seedFixedInd <- TRUE # fixed effects (true betas) and Sigma_x fixed across simulation iterations
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- 50 #50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- 1 #1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 50 # 0 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
rdgLmd <- TRUE # use the ridge hyperparamter from oseL0 to tune the ps2 and ps5 
lambdaZmax <- 5 # make sure lambda_z are smaller than this to pervent numerical instability
maxIter_train <- 5000
maxIter_cv <- 1000

tuneInd <- TRUE #sims6$tuneInd[runNum]
WSmethod <- 2 # sims6$WSmethod[runNum]
ASpass <- TRUE # sims6$ASpass[runNum]

if(tuneThreads == 1){
    # use non-parallel version
    source("sparseFn_iht_test_MT.R") # USE TEST VERSION HERE
    sparseCV_iht_par <- sparseCV_iht
}else{
    # source("sparseFn_iht_par.R")
    source("sparseFn_iht_test_MT.R") # USE TEST VERSION HERE
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
nfold <- 3
#oseTuneLong <- sims6$oseTn[runNum]

# maxRho: 20
# rho <- c(seq(2, 10, by = 2), seq(15, 20, by = 5))  

# maxRho: 50
rho <- c( seq(5, 15, by = 5), c(25, 25, 50) )

# c(seq(2, 10, by = 2), seq(15, 50, by = 5), seq(60, 90, by = 10)) # works great: 3-20-22 c( seq(5, 15, by = 5), c(25, 25, 50) ) #

#rho <- c( 50, seq(100, 325, by = 75)) # if ds is full marginal data then use this because dimension much higher
# rho <- c(seq(100, 225, by = 25)) # did well with this for breast cancer but added more for sparse 


rhoDiff <- round( mean( diff(rho) ) / 2 ) # average interval

lambda <- sort( unique( c(1e-6, 1e-5, 0.0001, 0.001, 0.01, 5,10, 50, 100,
                          exp(-seq(0,5, length = 10))
                          #seq(120, 200, by = 20)
                ) ), decreasing = TRUE ) # 2:100

lambdaShort <- sort( unique( c(
                               exp(-seq(0,5, length = 5)),
                               5,10, 50, 100, 250, 500, 1000, 2500, 5000, 10000) ),

                     decreasing = TRUE ) # 2:100

lambdaZ <- sort( unique( c(0, c( 10^c(-(6:3)), 5 * 10^c(-(6:3))),
                           exp(-seq(0,5, length = 8)),
                           1:3) ),
                 decreasing = TRUE ) # 2:100

lambdaZ <- lambdaZ * min(rho)

lambdaBeta <- c( 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 )


# file name
fileNm <- paste0("MT_breast",
                 "_K_", K,
                 "_study_", studies,
                 "_ds_", ds,
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
                 "_TnIn_", tuneInd)
print(fileNm)

timeStart <- Sys.time()

#####################################################################
print(paste0("start: ", iterNum))
    Sys.setenv(JULIA_BINDIR = juliaPath)

    ##############
    # MT versions
    ##############
    L0_reg <- juliaCall("include", paste0(juliaFnPath_MT, "l0_IHT_tune.jl") ) # sparseReg # MT: doesnt make sense
    L0_MS <- juliaCall("include", paste0(juliaFnPath_MT, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
    L0_MS2 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
    L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
    L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_tuneTest_MT.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
    L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study
    ####################################################

    resMat <- matrix(nrow = totalSims, ncol = 235)
    colnames(resMat) <- c("mergeRdg", "oseRdg","oseRdgStk",
                          "mrgL0", "mrgL0_fp", "mrgL0_tp",
                          "mrgL0_sup", "mrgL0_auc", "mrgL0Lrn",
                          "mrgL0Lrn_fp", "mrgL0Lrn_tp",
                          "mrgL0Lrn_sup", "mrgL0Lrn_auc",
                          "oseL0Lrn","oseL0LrnStk", "oseL0Lrn_fp",
                          "oseL0Lrn_tp", "oseL0Lrn_sup", "oseL0Lrn_auc",
                          "oseL0LrnStk_fp", "oseL0LrnStk_tp", "oseL0LrnStk_sup",
                          "oseL0LrnStk_auc", "studyL0Lrn_aucAvg", "oseL0",
                          "oseL0Stk", "oseL0_fp", "oseL0_tp", "oseL0_sup",
                          "oseL0_auc", "oseL0Stk_fp", "oseL0Stk_tp", "oseL0Stk_sup",
                          "oseL0Stk_auc", "studyL0_aucAvg",
                          "msP1_L0", "msP1_L0Stk", "msP1_L0_fp", "msP1_L0_tp",
                          "msP1_L0_sup", "msP1_L0_auc", "msP1_L0Stk_fp",
                          "msP1_L0Stk_tp", "msP1_L0Stk_sup", "msP1_L0Stk_auc",
                          "msP2_L0","msP2_L0Stk", "msP2_L0_fp", "msP2_L0_tp",
                          "msP2_L0_sup", "msP2_L0_auc", "msP2_L0Stk_fp",
                          "msP2_L0Stk_tp", "msP2_L0Stk_sup", "msP2_L0Stk_auc",
                          "msP1_con","msP1_conStk", "msP1_con_fp", "msP1_con_tp",
                          "msP1_con_sup", "msP1_con_auc", "msP1_conStk_fp",
                          "msP1_conStk_tp", "msP1_conStk_sup", "msP1_conStk_auc",
                          "msP4","msP4_Stk", "msP4_fp", "msP4_tp",
                          "msP4_sup", "msP4_auc", "ms4_Stk_fp",
                          "msP4_Stk_tp", "msP4_Stk_sup", "msP4_Stk_auc",
                          ####
                          "msP3_L0","msP3_L0Stk", "msP3_L0_fp", "msP3_L0_tp",
                          "msP3_L0_sup", "msP3_L0_auc", "msP3_L0Stk_fp",
                          "msP3_L0Stk_tp", "msP3_L0Stk_sup", "msP3_L0Stk_auc",
                          "msP3_con","msP3_conStk", "msP3_con_fp", "msP3_con_tp",
                          "msP3_con_sup", "msP3_con_auc", "msP3_conStk_fp",
                          "msP3_conStk_tp", "msP3_conStk_sup", "msP3_conStk_auc",
                          "msP1_L0_lambda", "msP2_L0_lambda", "msP1_con_lambda",
                          "msP4_lambda", "msP3_L0_lambda", "msP3_con_lambda",
                          "mergeLasso", "lassoMrg_fp", "lassoMrg_tp",
                          "lassoMrg_sup", "lassoMrg_auc",
                          ##
                          "MTL_L2L1_supp", "MTL_L2L1", "MTL_trace_supp",
                          "MTL_trace", 
                          ##
                          "mtl_ridge_r2", "MoM_mrg_rho",
                          ##
                          "MT_lasso_low", "MT_lasso_low_s",
                            ##
                          "MoM_mrg_L0_tp",
                          "MoM_mrg_L0_sup", "MoM_mrg_L0_auc", "MoM_mrg_L0_rho",

                          "MT_rmtl_lasso_low", "MT_rmtl_lasso_low_s", "MoM_fp", "MoM_tp",
                          "MoM_sup", "MoM_auc", "MoM_Stk_fp",
                          "MoM_Stk_tp", "MoM_Stk_sup", "MoM_Stk_auc", "MoM_aucAvg",
                          ##
                          "MoM_L0", "MoM_L0Stk", "MoM_L0_fp", "MoM_L0_tp",
                          "MoM_L0_sup", "MoM_L0_auc", "MoM_L0Stk_fp",
                          "MoM_L0Stk_tp", "MoM_L0Stk_sup", "MoM_L0Stk_auc", "Time",
                          ##
                          "lassoMrg_supPred", "L0LrnMrg_supPred", "L0Mrg_supPred", "oseL0Lrn_supPred",
                          "oseL0LrnStk_supPred", "oseL0_supPred", "oseL0Stk_supPred",
                          "PS1_supPred", "PS1stk_supPred", "PS2_supPred", "PS2Stk_supPred",
                          "PS4_supPred", "PS4stk_supPred",
                          "PS3_supPred", "PS3stk_supPred", "MoM_mrg_supPred", "MoM_L0mrg_supPred",
                          "MoM_supPred", "MoMstk_supPred",
                          "MoM_L0_supPred", "MoMStk_L0_supPred",
                          ##
                          "s_lassoMrg", "s_mrg_l0Lrn", "s_mrgL0", "s_oseL0Lrn", "s_oseL0LrnStk", "s_ose", "s_oseStk",
                          "s_MS1", "s_MS1Stk", "s_MS2", "s_MS2Stk", "s_MS1con", "s_MS1Stkcon","s_MS4", "s_MS4Stk", "s_MS3", "s_MS3Stk", 
                          "s_MS3con", "s_MS3conStk", "
                          s_MoM", "s_MoM_L0", "s_MoM_ose", "s_MoM_Stk","s_MoML0_ose", "s_MoML0_Stk",
                          ##
                          "msP5", "msP5_Stk", "msP5_fp", "msP5_tp", "msP5_sup", "msP5_auc", "ms5_Stk_fp",  
                          "msP5_Stk_tp", "msP5_Stk_sup", "msP5_Stk_auc", "PS5_supPred", "PS5stk_supPred", 
                          "msP5_lambda2", "msP5_lambda_z", "s_MS5", "s_MS5Stk",
                          ##
                          paste0("coef_",
                                 c("mtRdg", "mtLasso", "sseRdg", "L0LrnMrg", "L0Mrg", "oseL0Lrn", "oseL0", "msP1_L0",
                                   "msP2_L0", "msP1_con", "msP5_L0", "msP4_L0", "msP3_L0","msP3_con")),
                          ##
                          paste0("MT_",
                                 c("mtRdg", "mtLasso", "sseRdg", "L0LrnMrg", "L0Mrg", "oseL0Lrn", "oseL0", "msP1_L0",
                                   "msP2_L0", "msP1_con", "msP5_L0", "msP4_L0", "msP3_L0","msP3_con")
                                 
                          ), "time1", "time2", "time3", "totalTime", "timeRidge"
                          )


        # if true, then we fix the fixed effects (true betas) across iterations of simulation and just allow random effects to vary
        # also fixes Sigma
        if(seedFixedInd)   seedFixed <- 1 # arbitrarily set seed to 1 so fixed across iterations at seedFixed=1

        ################################
        # generate inclusion variables for support
        ################################
        # dummy variables
        trueB <- trueZ <- rep(1, numCovs)

        # split multi-study
        cnt <- seedSet <- studyNum <- iterNum
        set.seed(seedSet)
        
        splitSize <- 1 - (1 / totalSims) # number of cross validation 
        
        # if(Yscale)   full[,Yindx] <- scale(full[,Yindx]) # center and scale Ys
        
        set.seed(seedSet)
        mtTest <- full[full$Study == testStudy, -1] # test study, remove "Study" column
        full <- full[full$Study %in% trainStudies, -1] # all exceept test study, remove "Study" column
        sds <- apply(full, 2, sd) # standard deviation
        
        # remove variables which have sd == 0
        if( sum(sds == 0) > 0 ){
            indxSD <- which(sds == 0)
            full <- full[,-indxSD] # remove columns with sd = 0
            mtTest <- mtTest[,-indxSD] # remove columns with sd = 0
        }
        
        Xindx <- seq(K + 1, ncol(full) )
        Yindx <- 1:K
        numCovs <- length(Xindx)
        trueB <- trueZ <- rep(1, numCovs)
        
        ######################
        # make tuning grids
        ######################
        rho <- rho[ rho <= length(Xindx) ] # remove any rhos that are too large for number of features in dataset
        
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
        tune.gridGLM <- as.data.frame( cbind( 0, lambda) ) # Ridge
        
        colnames(tune.gridGLM) <- c("alpha", "lambda")

        ####################################
        # scale covariates and outcome
        ####################################
        nFull <- nrow( full ) # sample size of merged
        
        if(scaleInd == TRUE){
            
            # scale Covaraites
            means <- colMeans( as.matrix( full  ) )
            sds <- sqrt( apply( as.matrix( full  ), 2, var) *  (nFull - 1) / nFull )  # use mle formula to match with GLMNET
            
            # columns 1 and 2 are Study and Y
            for(column in Xindx ){
                # center scale
                full[, column] <- (full[, column ] - means[column]) / sds[column]
                mtTest[, column] <- (mtTest[, column ] - means[column ]) / sds[column]
                
            }
            
            
        }
        
        
        if(Yscale){
            
            sds_y <- sqrt( apply( as.matrix( full[,Yindx]  ), 2, var) *  (nFull - 1) / nFull )
            means_y <- colMeans( as.matrix( full[,Yindx]   ) )
            ii <- 1 # index
            
            for(yIndex in Yindx){
                
                full[,yIndex] <- (full[,yIndex] - means_y[ii]) / sds_y[ii]
                mtTest[,yIndex] <- (mtTest[,yIndex] - means_y[ii]) / sds_y[ii]
                ii <- ii + 1
                
            }
            
        }   
        
 
        rownames(mtTest) <- 1:nrow(mtTest)
        rownames(full) <- 1:nrow(full)

        timeStartTotal <- Sys.time()
       
        ##############
        # GLMNET Ridge - MultiTask
        ##############

        # replicate folds from tuning function
        allRows <- 1:nrow(full)
        set.seed(1)
        HOOL <- createFolds(allRows, k = nfold) # make folds 
        foldID <- vector( length = nrow(full) )
        for(fd in 1:nfold)  foldID[ HOOL[[fd]] ] <- fd
        
        tune.mod <- cv.glmnet(y = as.matrix(full[,Yindx]),
                              x = as.matrix(full[,Xindx]),
                              alpha = 0,# tune.gridGLM$alpha,
                              lambda = tune.gridGLM$lambda,
                              intercept = TRUE,
                              # parallel = TRUE,
                              foldid = foldID,
                              family = "mgaussian")
        
        betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = "lambda.min") ) )
        betaEst <- as.matrix( betaEst )
        
        resMat[iterNum, 217] <- multiTaskRmse_MT(data = mtTest, beta = betaEst)
        
        
        resMat[iterNum, 111] <- multiTaskR2_MT(data = mtTest, 
                                               Ytrain = as.matrix(full[,Yindx]), 
                                               beta = betaEst)
        
        rm(betaEst, tune.mod)
        
        ##############
        # GLMNET Lasso - MultiTask
        ##############
        tune.mod <- cv.glmnet(y = as.matrix(full[,Yindx]),
                              x = as.matrix(full[,Xindx]),
                              alpha = 1,# tune.gridGLM$alpha,
                              lambda = tune.gridGLM$lambda,
                              intercept = TRUE,
                              # parallel = TRUE,
                              foldid = foldID,
                              family = "mgaussian")
        
        lambSeq <- tune.mod$lambda # lambdas
        tn_len <- length(lambSeq) # length of tuning grid
        suppVec <- vector(length = tn_len)
        errorVec <- tune.mod$cvm
            
        for(j in 1:tn_len){
            beta_tn <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = lambSeq[j]) ) )
            suppVec[j] <- mean( apply(beta_tn, 2, function(x) sum(x[-1] != 0)) ) # avg sparsity
        }
        
        errorVec <- errorVec[ suppVec <= max(rho) ] # errors for which rho is smaller than the max
        sV <- suppVec[ suppVec <= max(rho) ]
        lambs <- lambSeq[ suppVec <= max(rho) ]
        
        lambdaStar <- lambs[which.min(errorVec)]
        
        betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = "lambda.min") ) )
        betaEst <- as.matrix(betaEst)

        resMat[iterNum, 218] <- multiTaskRmse_MT(data = mtTest, beta = betaEst)
        resMat[iterNum, 162] <- sum( betaEst[-1,] != 0 ) / K # size of xupport

        rm(betaEst)
        
        # best lasso with s rho constraint
        betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = lambdaStar) ) )
        betaEst <- as.matrix(betaEst)
        
        resMat[iterNum, 113] <- multiTaskRmse_MT(data = mtTest, beta = betaEst)
        resMat[iterNum, 114] <- sum( betaEst[-1,] != 0 ) / K # size of xupport
        # # 218
        rm(betaEst, tune.mod, lambdaStar, errorVec, suppVec, lambSeq)


        ##############
        # OSE L0Learn
        ##############
        print(paste("iteration: ", iterNum, " L0Learn OSE"))
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        # testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        betaMat <- matrix(NA, ncol = K, nrow = length(Xindx) + 1) # matrix of betas from each study
        res <- vector(length = K) # store auc
        Mvec <- vector(length = K) # use to initialize other models later on
        # save best parameter values
        L0_tune <- matrix(NA, nrow = K, ncol = ncol(tune.grid) ) # save best parameter values
        L0_tune <- as.data.frame(L0_tune)
        colnames(L0_tune) <- colnames(tune.grid)

        supMat <- matrix(NA, nrow = K, ncol = 4)

        for(j in 1:K){
            print(j)
           # indx <- which(full$Study == j) # rows of each study
        #    n_k <- length(indx) # sample size of jth study
            sdY <- 1 # set to 1 for now so we DO NOT adjust as glmnet() #sd(full$Y[indx]) * (n_k - 1) / n_k #MLE
            gm <- tune.grid$lambda / (sdY * 2) # convert into comparable numbers for L0Learn

            # fit l0 model on jth study
            cvfit = L0Learn.cvfit(x = as.matrix(full[, Xindx]),
                                  y = as.vector(full[,j]),
                                  nFolds = nfold, # caused problems for low n_k settings
                                  seed = 1,
                                  penalty="L0L2",
                                  nGamma = length(gm),
                                  #gammaMin = min(gm), # min and max numbers of our 2 parameter that is comaprable
                                  #gammaMax = max(gm),
                                  algorithm = "CD", #"CDPSI",
                                  maxSuppSize = max(tune.grid$rho)#, # largest that we search
                                  # scaleDownFactor = 0.99
            )
            
            # optimal tuning parameters
            optimalGammaIndex <- which.min( lapply(cvfit$cvMeans, min) ) # index of the optimal gamma identified previously
            optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
            optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
            L0LearnCoef <- coef(cvfit, lambda=optimalLambda, gamma = cvfit$fit$gamma[optimalGammaIndex] )

            # save tuned parameter values
            rhoStar <- sum(  as.vector(L0LearnCoef)[-1] != 0   ) # cardinality
            L0_tune$lambda[j] <- cvfit$fit$gamma[optimalGammaIndex] * (2 * sdY) # put on scale used by gurobi version below
            L0_tune$rho[j] <- rhoStar

            # use L0Learn coefficients as warm starts
            betaMat[,j] <- L0LearnCoef <- as.vector(L0LearnCoef) # save coefficients -- use "betas" as warm start for later

            # stack matrix
            predsMat[,j] <- cbind( 1, as.matrix( full[,Xindx ] ) ) %*% L0LearnCoef

            rm(cvfit)
        }

        rm(sdY)

        zAvg <- I( rowMeans(betaMat)[-1] != 0) * 1 # stacking

        resMat[iterNum, 165] <- mean(L0_tune$rho) # cardinality of support
        resMat[iterNum, 166] <- sum( zAvg ) # cardinality of support
        resMat[iterNum, 222] <- multiTaskRmse_MT(data = mtTest, beta = betaMat)
        
        rm(betaMat, res, zAvg)

        
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
        
        tune.grid_OSE <- data.frame(lambda1 = unique(lambda),
                                    lambda2 = 0,
                                    lambda_z = 0,
                                    #rho = numCovs
                                    rho = tune.grid$rho)
        
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
        
        rm(L0_MS_z3)
        L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study
        
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
        
        resMat[iterNum, 233] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        
        resMat[iterNum, 167] <- L0_tune$rho # cardinality of support
        resMat[iterNum, 223] <- multiTaskRmse_MT(data = mtTest, beta = betas)
        
        rm(res, L0_tune, betas)

        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################
        # common support L0 regularization with ||beta - betaBar|| penalty
        # glPenalty = TRUE, ip = TRUE
        ####################################
        # random initialization for betas
        #betas <- matrix( 0, nrow = numCovs + 1, ncol = K )#matrix( rnorm( K * p ), ncol = K )

        print(paste("iteration: ", iterNum, " Group IP"))
        glPenalty <- 1
        ip <- TRUE

        # tune multi-study with l0 penalty with GL Penalty = TRUE

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

        rm(L0_MS2)
        L0_MS2 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
        
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

        resMat[iterNum, 96] <- MSparams$lambda2 # tuning parameter
        
        resMat[iterNum, 169] <- MSparams$rho # cardinality of support

        resMat[iterNum, 224] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS, warmStart)

        ########################################################
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################
        #  L0 regularization with ||z - zbar|| penalty and (potentially) ||beta - betaBar|| and (potentially) frobenius norm
        # glPenalty = 2
        ####################################
        timeStart1 <- Sys.time()
        
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        # testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        res <- resS <- vector(length = K) # store support prediction
        
        if(tuneInd){
            
            # ************
            # original from 1/20/21
            # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
            # ************
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( ridgeLambda, 0, lambdaZ, rho) )
            # ridgeLambda * (rho / ridgeRho)
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
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
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            # rhoG <- seq(rhoStar - rhoDiff, rhoStar + rhoDiff, length = 3)
            # rhoG <- sapply(rhoG, function(x) max(x, 2 ) ) # make sure there is nothing below 2
            # lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- c( seq(1.5, 10, length = 5), seq(0.1, 1, length = 5) ) * lambdaZstar
            lambdaZgrid <- lambdaZgrid[lambdaZgrid <= lambdaZmax] # make sure this is below a threshold to prevent numerical issues
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # original from 1/20/21
            # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
            # ************
            gridUpdate <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZgrid, rhoStar) )
            colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                             gridUpdate$lambda1,
                                             -gridUpdate$lambda_z,
                                             decreasing=TRUE),     ]
            
            gridUpdate <- unique(gridUpdate)
            
            lambdaZScale <- rhoScale(K = K, 
                                     p = numCovs, 
                                     rhoVec = gridUpdate$rho, 
                                     itrs = 100000,
                                     seed = 1)
            
            gridUpdate <- tuneZscale(tune.grid = gridUpdate, 
                                          rhoScale = lambdaZScale)
            
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = gridUpdate,
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
            
        }else{
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZ, rho) )
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
        }
        
        MSparams <- tuneMS$best # parameters

        rm(L0_MS_z)
        L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
        
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
        
        multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        timeEnd1 <- timeEnd <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 231] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        
        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,Xindx ] ) ) %*% betasMS
 
        rm(predsMat)

        resMat[iterNum, 97] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 171] <- MSparams$rho # cardinality of support

        resMat[iterNum, 225] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, res, resS, betasMS, warmStart )
        ########################################################
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################################
        # convex version of MS (no cardinality constraints)
        ####################################################
        print(paste("iteration: ", iterNum, " Group Convex"))
        ####################################
        # ||beta - betaBar|| and NO Frobenius norm no cardinality constraints
        # Convex, IP = FALSE
        ####################################
        # glPenalty <- 1
        # ip <- FALSE
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
        L0_MS2 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_tune_MT.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
        
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


        resMat[iterNum, 98] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 173] <- MSparams$rho # cardinality of support

        resMat[iterNum, 226] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS)
        ########################################################
        #
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################
        # MS5 -- different support with beta - betaBar penalty AND z- zbar penalty, no frobenius
        ####################################
        # share info on the beta - betaBar AND ||z - zbar|| 
        timeStart1 <- Sys.time()
        
        glPenalty <- 4
        
        if(tuneInd){
            
            # ************
            # original from 1/20/21
            # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
            # ************
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0.01, 0, lambdaZ, rho) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
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
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            # rhoG <- seq(rhoStar - rhoDiff, rhoStar + rhoDiff, length = 3)
            # rhoG <- sapply(rhoG, function(x) max(x, 2 ) ) # make sure there is nothing below 2
            # lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- c( seq(1.5, 10, length = 5), seq(0.1, 1, length = 5) ) * lambdaZstar
            lambdaZgrid <- lambdaZgrid[lambdaZgrid <= lambdaZmax] # make sure this is below a threshold to prevent numerical issues
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # original from 1/20/21
            # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
            # ************
            gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBeta, lambdaZgrid, rhoStar) )
            colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                             -gridUpdate$lambda2,
                                             -gridUpdate$lambda_z,
                                             decreasing=TRUE),     ]
            
            gridUpdate <- unique(gridUpdate)
            
            lambdaZScale <- rhoScale(K = K, 
                                     p = numCovs, 
                                     rhoVec = gridUpdate$rho, 
                                     itrs = 100000,
                                     seed = 1)
            
            gridUpdate <- tuneZscale(tune.grid = gridUpdate, 
                                          rhoScale = lambdaZScale)
            
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = gridUpdate,
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
            
        }else{
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, lambdaBeta, lambdaZ, rho) )
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
        }
        
        MSparams <- tuneMS$best # parameters
        ###################################################
        rm(L0_MS_z)
        L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
        
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
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 232] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))

        resMat[iterNum, 199] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 200] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 201] <- MSparams$rho # cardinality of support
        resMat[iterNum, 227] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS, warmStart)
        
        ########################################################
        # ******************************************************
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################
        ####################################
        # MS4 -- different support with beta - betaBar penalty AND NO frobenius norm AND NO z- zbar penalty
        ####################################
        # share info on the beta - betaBar but no ||z - zbar|| penalty (currently but could do it if changed tuning grid)
        glPenalty <- 4

        # *************************
        # original 1/20/22
        if(tuneInd){
            
            tune.grid_beta <- as.data.frame(  expand.grid( 0, lambdaBeta, 0, rho) ) # tuning parameters to consider
            colnames(tune.grid_beta) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_beta <- tune.grid_beta[  order(-tune.grid_beta$rho,
                                                     -tune.grid_beta$lambda2,
                                                       decreasing=TRUE),     ]
            
            tune.grid_beta <- unique(tune.grid_beta)
            # if tune in two stages
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = tune.grid_beta,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = TRUE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       maxIter = maxIter_cv,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaBstar <- MSparams$lambda2
            
            # rhoG <- seq(rhoStar - rhoDiff, rhoStar + rhoDiff, length = 3)
            # rhoG <- sapply(rhoG, function(x) max(x, 2 ) )
            # lambdaBgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaBstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaBgrid<- c( seq(1.5, 10, length = 5), seq(0.1, 1, length = 5) ) * lambdaBstar 

            gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBgrid, 0, rhoStar) )
            colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                             -gridUpdate$lambda2,
                                             -gridUpdate$lambda_z,
                                             decreasing=TRUE),     ]
            
            gridUpdate <- unique(gridUpdate)
            
            tuneMS <- sparseCV_iht_par(data = full,
                                       tune.grid = gridUpdate,
                                       hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                       method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                                       nfolds = nfold,
                                       cvFolds = 5,
                                       juliaPath = juliaPath,
                                       juliaFnPath = juliaFnPath,
                                       messageInd = TRUE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       maxIter = maxIter_cv,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
        }else{
            # tune in 1 stage
            
            tune.grid_beta <- as.data.frame(  expand.grid( 0, lambdaBeta, 0, rho) ) # tuning parameters to consider
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
        }
        
        MSparams <- tuneMS$best # tuned parameters
        
        rm(L0_MS_z)
        L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
        
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
        
        resMat[iterNum, 99] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 175] <- MSparams$rho # cardinality of support

        resMat[iterNum, 228] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS, warmStart)
        ########################################################
        # ******************************************************
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
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
        # testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set

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
        L0_MS <- juliaCall("include", paste0(juliaFnPath_MT, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
        
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

        resMat[iterNum, 100] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 177] <- MSparams$rho # cardinality of support

        resMat[iterNum, 229] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS, warmStart)
        ########################################################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        ####################################
        # Convex MS, glPenalty = 3, IP FALSE: Just Frobenius norm and no other sharing of information
        ####################################
        glPenalty <- 3
        ip <- FALSE
        tune.grid2 <- tune.grid
        tune.grid2$rho <- length(Xindx) # full support
        tune.grid2 <- unique(tune.grid2)
        #betas <- matrix(0, ncol = K, nrow = numCovs + 1)
        # testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set

        # tune multi-study with l0 penalty with GL Penalty = TRUE

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
        L0_MS <- juliaCall("include", paste0(juliaFnPath_MT, "BlockIHT_tune_MT.jl") ) # MT: Need to check it works
        
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

        resMat[iterNum, 101] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 179] <- MSparams$rho # cardinality of support
        #resMat[iterNum, 180] <- sum( zStack ) # cardinality of support
        #resMat[iterNum, 216] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 230] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS)
        ########################################################
        ########################################################

        ##########################
        # save results before MoM
        ##########################
        print("setWD to save BEFORE MoM")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)

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
            Xlist[[kk]] <- cbind(1, full[,Xindx]) # same design matrix for all
            yind <- Yindx[kk]
            Ylist[[kk]] <- full[,yind]
        }
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="L21", 
                        nfolds = nfold, 
                        #Lam1_seq = lambda,
                        Lam2 = 0) #, #lambda,
                        # parallel=TRUE, 
                        # ncores = 3)
        
        model <- MTL(X = Xlist, 
                     Y = Ylist, 
                     type="Regression", 
                     Regularization="L21",
                     Lam1=cvfitc$Lam1.min, 
                     Lam2 = 0, #cvfitc$Lam2.min, 10^seq(1,-4, -1)
                     opts=list(init=0,  tol=10^-6,
                               maxIter=1500), 
                     Lam1_seq=cvfitc$Lam1_seq
        )
        
        # FIX INDICES "mtl_L2L1_supp"
        
        # FIX INDICES "mtl_L2L1_supp"
        
        resMat[iterNum, 107] <-  S <- sum( model$W[-1,] != 0 ) / K # size of support
        resMat[iterNum, 108] <- multiTaskRmse(data = mtTest, beta = model$W)
        
        
        rm(model)
        
        ##########################################
        # only run if solution isn't sparse enough
        if( S > max(rho) ){
            
            lambda_vector <- sort( unique( c(0.0001, 0.001, 0.01, 5,10, 50, 100, 200,
                                             exp(-seq(0,5, length = 15))) ), 
                                   decreasing = TRUE ) 
            
            cvfitc <- cvMTL(X = Xlist, 
                            Y = Ylist,
                            type="Regression", 
                            Regularization="L21", 
                            nfolds = nfold, 
                            Lam1_seq = lambda_vector, # run with larger lambda vector
                            Lam2 = 0) #, #lambda,
                            # parallel=TRUE)
            
            lambSeq <- cvfitc$Lam1_seq
            len <- length(lambSeq)
            errorVec <- cvfitc$cvm
            
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
            
            errorVec <- errorVec[ suppVec <= max(rho) ] # errors for which rho is smaller than the max
            sV <- suppVec[ suppVec <= max(rho) ]
            lambs <- lambSeq[ suppVec <= max(rho) ]
            
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
                         Lam1_seq=cvfitc$Lam1_seq
            )
            
            # FIX INDICES "mtl_L2L1_supp"
            
            resMat[iterNum, 119] <-  sum( model$W[-1,] != 0 ) / K # size of wupport
            resMat[iterNum, 120] <- multiTaskRmse(data = mtTest, beta = model$W)
            rm(model, cvfitc)
            
        }else{
            
            # if previous solutions are sparse enough, use them
            resMat[iterNum, 111] <-  resMat[iterNum, 107]
            resMat[iterNum, 112] <- resMat[iterNum, 108]
            
        }
        
        ##############
        # RMTL - Trace
        ##############
        # format data for multi-task learning
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="Trace", 
                        #Lam1_seq = lambda,
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
        
        # FIX INDICES
        resMat[iterNum, 109] <-  sum( model$W[-1,] != 0 ) / K # size of xupport
        resMat[iterNum, 110] <- multiTaskRmse_MT(data = mtTest, beta = model$W)
        
        rm(model, cvfitc, Xlist, Ylist)
        ########################################################
        print(paste("iteration: ", iterNum, " Complete!"))
        print(resMat[iterNum,])
        
        # time difference
        timeEnd <- Sys.time()
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 140] <- as.numeric(difftime(timeEnd, timeStart, units='mins'))

        timeEndTotal <- Sys.time()
        resMat[iterNum, 234] <- as.numeric(difftime(timeEndTotal, timeStartTotal, units='mins'))
        ########################
        # save results
        ########################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        
        #####################################################################
