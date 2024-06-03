library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)
library(car)
library(pROC)
library(sparsegl)
library(grpreg)
library(MASS)
library(RMTL)
library(Matrix)
library(mixtools)

cluserInd <- TRUE # whether running on computer or on cluster

if(cluserInd){
  
  # this R code is saved in: /home/loewingergc/smtl/code/
  # bash in: /home/loewingergc/bash
  # parameters sims6 in: /home/loewingergc//smtl/data
  
  # paths
  wd <- "/home/loewingergc/smtl/code/" 
  data_path <-"/home/loewingergc/smtl/data" # path to original Matlab files for data
  save.folder <- "/home/loewingergc/smtl/fscv" # simulation_scienc
  
  
  #################
  # if on cluster
  args = commandArgs(TRUE)
  n_k <- as.integer( as.numeric(args[1]) ) # sample size to subsample
  colSubSamp <- as.integer( as.numeric(args[2]) )
  iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id
  paste("colSubSamp", colSubSamp)
  
  juliaPath <- "/usr/local/apps/julialang/1.7.3/bin" #"/usr/local/lmod/modulefiles"
  juliaFnPath_MT <- juliaFnPath <- "/home/loewingergc/smtl/code/"
  
}else{
  runNum <- 1
  setwd("~/Desktop/Research")
  iterNum <- 1
  n_k <- 100
  colSubSamp <- 10 # sample every colSubSamp^th row (so = 1 means use all of them and 2 samples every other)
  data_path <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/fscv"
  # Julia paths
  juliaPath = "/Applications/Julia-1.7.app/Contents/Resources/julia/bin"
  juliaFnPath_MT <- juliaFnPath <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/IHT/Tune/"
  save.folder <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study"
  wd <- "/Users/loewingergc/Downloads/sMTL-main/sMTL_Paper/sMTL_Functions/"
}

Sys.setenv(JULIA_BINDIR = juliaPath)

# parameters
K <- totalStudies <- 4
K_train <- K # number of training studies to randomly select
Yscale <- TRUE # scale outcome
totalSims <- 100
scaleInd <- TRUE
outcome <- "DA" # which is used as an outcome
seedFixedInd <- TRUE # fixed effects (true betas) and Sigma_x fixed across simulation iterations
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- NA # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- NA# when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 10# number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
rdgLmd <- TRUE # use the ridge hyperparamter from oseL0 to tune the ps2 and ps5 
lambdaZmax <- 5 # make sure lambda_z are smaller than this to pervent numerical instability
dataPro <- 2
Yscale <- TRUE
tune_length <- 100 # number of parameters to tune over to keep constant across methods
maxIter_train <- 5000
maxIter_cv <- 1000
tuneInd <- TRUE 
WSmethod <- 2 
ASpass <- TRUE 

# use non-parallel version
source(paste0(wd, "sparseFn_iht_test_MT.R")) # USE TEST VERSION HERE
source(paste0(wd, "sparseFn_iht_test.R")) # USE TEST VERSION HERE
sparseCV_iht_par <- sparseCV_iht

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

rho <- 25
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
lambda <- sort( unique( c(exp(seq(0, 6, length = tune_length/2)),
                          exp(-seq(0, 25, length = tune_length/2)) ) ), decreasing = TRUE ) 

lambdaZ <- sort( unique( c(0, 
                           exp(seq(0,4.75, length = round(tune_length/2) - 11)),
                           exp(-seq(0,25, length = round(tune_length/2) - 11))) ),
                 decreasing = TRUE ) 

# file name
fileNm <- paste0("cfFSCV_MT", 
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

resMat <- matrix(nrow = totalSims, ncol = 307)
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
                      "MT_lasso_low", "MT_lasso_low_s",
                      ##
                      "MT_oseL0_lambdaL2", "MT_msP2_lambdaL2", "MoM_mrg_L0_tp",
                      "MoM_mrg_L0_sup", "MoM_mrg_L0_auc", "MoM_mrg_L0_rho",
                      "MoM", "MoM_Stk", "MoM_fp", "MoM_tp",
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
                             c("mrgRdg", "mrgLasso", "sseRdg", "L0LrnMrg", "L0Mrg", "oseL0Lrn", "oseL0", "msP1_L0",
                               "msP2_L0", "msP1_con", "msP5_L0", "msP4_L0", "msP3_L0","msP3_con")),
                      ##
                      paste0("MT_",
                             c("mrgRdg", "mrgLasso", "sseRdg", "L0LrnMrg", "L0Mrg", "oseL0Lrn", "oseL0", "msP1_L0",
                               "msP2_L0", "msP1_con", "msP5_L0", "msP4_L0", "msP3_L0","msP3_con")
                             
                      ), "time1", "time2", "time3", "totalTime", "timeRidge",
                      paste0("SGL_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("SGLun_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("GL_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("GLun_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("grMCP_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("gel_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("cMCP_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                      paste0("grLS_",
                             c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")) )

        
        seedSet <- iterNum # ensures no repeats
        set.seed(seedSet)

        # data
        full <- read.csv(paste0(data_path, "/sub_samp2500")) #read.csv( paste0("/n/home12/gloewinger/fscv_subset") )
        full <- subset(full, select=-c(X.2, X.1, X, Channel, pH)) # remove unwanted variables
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
        
        nFull <- nrow(full)
        rownames(mtTest) <- 1:nrow(mtTest)
        rownames(full) <- 1:nFull
        
        # tuning grids
        rho <- rho[ rho < length(Xindx) ] # remove any rhos that are too large for number of features in dataset
        tune.grid_MS2 <- as.data.frame(  expand.grid( 0, lambda, rho) ) # tuning parameters to consider
        tune.grid_MSZ <- as.data.frame(  expand.grid( lambda, 0, lambdaZ, rho) ) # tuning parameters to consider
        tune.grid_MSZ_2 <- as.data.frame(  expand.grid( 0, lambda, lambdaZ, rho) ) # tuning parameters to consider
        tune.grid_MSZ_3 <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
        tune.grid_beta <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
        
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
        
        tune.grid <- as.data.frame(  expand.grid(c(lambda) , rho) ) # tuning parameters to consider
        colnames(tune.grid) <- c("lambda", "rho")
        
        # glmnet for ridge
        tune.gridGLM <- as.data.frame( cbind( 0, lambda) ) # Ridge
        colnames(tune.gridGLM) <- c("alpha", "lambda")
        
        timeStartTotal <- Sys.time()
        
        ############
        # OSE L0
        ############
        timeStart1 <- Sys.time()
        b <- matrix(0, ncol = K, nrow = numCovs + 1)
        
        tune.grid_OSE <- expand.grid(lambda1 = unique(lambda),
                                    lambda2 = 0,
                                    lambda_z = 0,
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
                                    threads = tuneThreads,
                                    WSmethod = WSmethod,
                                    ASpass = ASpass)
        
        L0_tune <- L0_tune$best # parameters
        ridgeRho <- L0_tune$rho # save for later for z - zBar penalties
        ridgeLambda <- L0_tune$lambda1 # save for later for z - zBar penalties
        
        # initialize algorithm warm start
        p <- ncol(full) - 1
        b <-  matrix(0, ncol = K, nrow = p ) #rep(0, p) #rnorm( p )
        
        # matrices
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- vector(length = K) # store auc
        supMat <- matrix(NA, nrow = K, ncol = 4)
        
        # warm start
        warmStart = L0_MS_z3(X = as.matrix( full[ , -c(1,2) ]) ,
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
                             ASpass = ASpass)
        
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
                         ASpass = ASpass)
        
        timeEnd1 <- Sys.time()
        resMat[iterNum, 233] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        resMat[iterNum, 167] <- L0_tune$rho # cardinality of support
        resMat[iterNum, 223] <- multiTaskRmse(data = mtTest, beta = betas)
        rm(res, L0_tune)
        
        ####################################
        #  ZBar + L2 
        ####################################
        timeStart1 <- Sys.time()
        b <- matrix(0, ncol = K, nrow = numCovs + 1)
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- resS <- vector(length = K) # store support prediction
        
        if(tuneInd){
          
          tune.grid_MSZ_5 <- as.data.frame(  expand.grid( ridgeLambda / 2, 0, lambdaZ, rho) ) 
          colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
          tune.grid_MSZ_5$lambda1 <- tune.grid_MSZ_5$lambda1 * (tune.grid_MSZ_5$rho / ridgeRho)
          
          # order correctly
          tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                     -tune.grid_MSZ_5$lambda_z,
                                                     decreasing=TRUE),     ]
          tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
          
          # tune z - zbar and rho
          tuneMS <- sparseCV_iht_par(data = full,
                                     tune.grid = tune.grid_MSZ_5,
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
                                     WSmethod = WSmethod,
                                     ASpass = ASpass )
          
          MSparams <- tuneMS$best # parameters
          rhoStar <- MSparams$rho
          lambdaZstar <- MSparams$lambda_z
          
          rhoG <- rho 
          lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
          lambdaZgrid <- sort( unique(lambdaZgrid), decreasing = TRUE)
          
          gridUpdate <- as.data.frame(  expand.grid( ridgeLambda / 2, 0, lambdaZgrid, rhoG) )
          colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                           gridUpdate$lambda1,
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
                                     WSmethod = WSmethod,
                                     ASpass = ASpass)
          
        }else{
          
          tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZ, rho) )
          colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          tune.grid_MSZ_5$lambda1 <- tune.grid_MSZ_5$lambda1 * (tune.grid_MSZ_5$rho / ridgeRho)
          
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
                                     WSmethod = WSmethod,
                                     ASpass = ASpass)
        }
        
        MSparams <- tuneMS$best # parameters
        
        # warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
        warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                            y = as.vector(full$Y),
                            rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = b,
                            lambda1 = MSparams$lambda1,
                            lambda2 = MSparams$lambda2,
                            lambda_z = 0,
                            scale = TRUE,
                            maxIter = 10000,
                            localIter = 0,
                            WSmethod = WSmethod,
                            ASpass = ASpass)
        
        # final model
        betasMS = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                          y = as.vector(full$Y),
                          rho = MSparams$rho,
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = warmStart,
                          lambda1 = MSparams$lambda1,
                          lambda2 = MSparams$lambda2,
                          lambda_z = MSparams$lambda_z,
                          scale = TRUE,
                          maxIter = 10000,
                          localIter = 50,
                          WSmethod = WSmethod,
                          ASpass = ASpass)
        
        timeEnd1 <- timeEnd <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 231] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        resMat[iterNum, 97] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 171] <- MSparams$rho # cardinality of support
        resMat[iterNum, 114] <- MSparams$lambda1
        resMat[iterNum, 225] <- multiTaskRmse(data = mtTest, beta = betasMS)
        rm(MSparams, tuneMS, betasMS,warmStart )
        ########################################################
        
        ####################################
        # MS4 -- Bbar
        ####################################
        tune.grid_beta <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
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
                                   ASpass = ASpass)
        
        MSparams <- tuneMS$best # parameters
        rhoStar <- MSparams$rho
        lambdaBstar <- MSparams$lambda2
        
        rhoG <- rho 
        lambdaBgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaBstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
        lambdaBgrid <- sort(lambdaBgrid, decreasing = TRUE)
        
        gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBgrid, 0, rhoG) )
        colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
        
        gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                         -gridUpdate$lambda2,
                                         -gridUpdate$lambda_z,
                                         decreasing=TRUE),     ]
        
        gridUpdate <- unique(gridUpdate)
        
        # tune multi-study with l0 penalty with only beta - betaBar penalties
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
                                   ASpass = ASpass)
        
        MSparams <- tuneMS$best # tuned parameters
        
        # warm start
        warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                            y = as.matrix( full[, Yindx] ),
                            rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = b,
                            lambda1 = max(MSparams$lambda1, 1e-5), # speed up solution
                            lambda2 = 0, 
                            lambda_z = 0, 
                            scale = TRUE,
                            maxIter = maxIter_train,
                            localIter = 0,
                            WSmethod = WSmethod,
                            ASpass = ASpass)
        
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
                          ASpass = ASpass)
        
        resMat[iterNum, 99] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 175] <- MSparams$rho # cardinality of support
        resMat[iterNum, 228] <- multiTaskRmse(data = mtTest, beta = betasMS)
        rm(MSparams, tuneMS, betasMS, warmStart)
        ########################################################
        
        ####################################
        # MS5 -- betaBar AND zbar
        ####################################
        timeStart1 <- Sys.time()
        
        if(tuneInd){

          lambdaZstar <- resMat[iterNum, 97] # use tuned value above to get into right ball park
          tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, lambda, lambdaZstar, rho) ) 
          colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          # order correctly
          tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                     -tune.grid_MSZ_5$lambda_z,
                                                     decreasing=TRUE),     ]
          
          tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
          
          print(paste("iteration: ", iterNum, " zBar/bBar Tune 1"))
          
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
                                     messageInd = TRUE,
                                     LSitr = LSitr, 
                                     LSspc = LSspc,
                                     maxIter = maxIter_cv,
                                     threads = tuneThreads,
                                     WSmethod = WSmethod,
                                     ASpass = ASpass)
          
          MSparams <- tuneMS$best # parameters
          rhoStar <- MSparams$rho
          lambdaZstar <- MSparams$lambda_z
          lambdaBstar <- MSparams$lambda2
          
          rhoG <- rho 
          lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
          lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)

          gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBstar, lambdaZgrid, rhoG) ) 
          colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                           -gridUpdate$lambda2,
                                           -gridUpdate$lambda_z,
                                           decreasing=TRUE),     ]
          
          gridUpdate <- unique(gridUpdate)
          
          print(paste("iteration: ", iterNum, " zBar/bBar Tune 2"))
          
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
                                     maxIter = maxIter_cv,
                                     threads = tuneThreads,
                                     WSmethod = WSmethod,
                                     ASpass = ASpass)
            
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
                                       WSmethod = WSmethod,
                                       ASpass = ASpass )
        }
        ###################################################
        MSparams <- tuneMS$best # parameters

        # warm start
        warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                            y = as.vector(full$Y),
                            rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = b,
                            lambda1 = MSparams$lambda1,
                            lambda2 = 0, 
                            lambda_z = MSparams$lambda_z,
                            scale = TRUE,
                            maxIter = 10000,
                            localIter = 0,
                            WSmethod = WSmethod,
                            ASpass = ASpass )
        
        # final model
        betasMS = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                          y = as.vector(full$Y),
                          rho = MSparams$rho,
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = warmStart,
                          lambda1 = MSparams$lambda1,
                          lambda2 = MSparams$lambda2,
                          lambda_z = MSparams$lambda_z,
                          scale = TRUE,
                          maxIter = 10000,
                          localIter = 50 ,
                          WSmethod = WSmethod,
                          ASpass = ASpass)
        
        timeEnd1 <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 232] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        resMat[iterNum, 199] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 200] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 201] <- MSparams$rho # cardinality of support
        resMat[iterNum, 227] <- multiTaskRmse(data = mtTest, beta = betasMS)
        rm(MSparams, tuneMS, betasMS, warmStart)

        ####################################
        # common support L0 regularization with ||beta - betaBar|| penalty
        # glPenalty = TRUE, ip = TRUE
        ####################################
        print(paste("iteration: ", iterNum, " Group IP"))
        tune.grid_MS2 <- as.data.frame(  expand.grid( 0, lambda, rho) ) # tuning parameters to consider
        colnames(tune.grid_MS2) <- c("lambda1", "lambda2", "rho")
        
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
                                   LSitr = LSitr,
                                   LSspc = LSspc,
                                   threads = tuneThreads)
        
        MSparams <- tuneMS$best # parameters
        
        # warm start for 4 * rho and no betaBar regularization
        warmStart = L0_MS2(X = as.matrix( full[ , -c(1,2) ] ),
                           y = as.vector(full$Y),
                           rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                           study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                           beta = b,
                           lambda1 = MSparams$lambda1,
                           lambda2 = 0, # use 0 as warm start
                           scale = TRUE,
                           maxIter = 10000,
                           localIter = 0)
        
        # final model
        betasMS = L0_MS2(X = as.matrix( full[ , -c(1,2) ]) ,
                         y = as.vector(full$Y),
                         rho = MSparams$rho,
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = warmStart,
                         lambda1 = MSparams$lambda1,
                         lambda2 = MSparams$lambda2,
                         scale = TRUE,
                         maxIter = 10000,
                         localIter = 50)
 
        resMat[iterNum, 96] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 169] <- MSparams$rho # cardinality of support
        resMat[iterNum, 224] <- multiTaskRmse(data = mtTest, beta = betasMS)
        rm(MSparams, tuneMS, betasMS, warmStart)
        
        ########################################################
        print(paste("iteration: ", iterNum, " Group Convex"))
        ####################################
        # ||beta - betaBar|| and NO Frobenius norm no cardinality constraints
        # Convex
        ####################################
        tune.grid_MS2 <- as.data.frame(  expand.grid( 0, lambda, ncol(full) - 2) ) # tuning parameters to consider
        colnames(tune.grid_MS2) <- c("lambda1", "lambda2", "rho")
        
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
                                   threads = tuneThreads )
        
        MSparams <- tuneMS$best # parameters
        
        betasMS = L0_MS2(X = as.matrix( full[ , -c(1,2) ]) ,
                         y = as.vector(full$Y),
                         rho = MSparams$rho,
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = b,
                         lambda1 = MSparams$lambda1,
                         lambda2 = MSparams$lambda2,
                         scale = TRUE,
                         maxIter = 10000,
                         localIter = 0) # convex
 
        resMat[iterNum, 98] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 173] <- MSparams$rho # cardinality of support
        resMat[iterNum, 226] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS)

        ####################################
        # MS3 - common support L0 regularization with L2
        ####################################
  
        tune.grid <- as.data.frame(  expand.grid( c(lambda), rho) ) 
        colnames(tune.grid) <- c("lambda", "rho")
        
        # tune multi-study with l0 penalty with GL Penalty = TRUE
        tuneMS <- sparseCV_iht_par(data = full,
                                   tune.grid = tune.grid,
                                   hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                   method = "MS", # could be L0 for sparse regression or MS # for multi study
                                   nfolds = nfold,
                                   cvFolds = 5,
                                   juliaPath = juliaPath,
                                   juliaFnPath = juliaFnPath,
                                   messageInd = FALSE,
                                   LSitr = LSitr,
                                   LSspc = LSspc,
                                   threads = tuneThreads)
        
        MSparams <- tuneMS$best # parameters
        
        # warm start
        warmStart = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                          y = as.vector(full$Y),
                          rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = b,
                          lambda = MSparams$lambda,
                          scale = TRUE,
                          maxIter = 10000,
                          localIter = 0)
        
        # final model
        betasMS = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                        y = as.vector(full$Y),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = warmStart,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 50)
        
        resMat[iterNum, 100] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 177] <- MSparams$rho # cardinality of support
        resMat[iterNum, 229] <- multiTaskRmse(data = mtTest, beta = betasMS)
        rm(MSparams, tuneMS, betasMS, warmStart)
        ########################################################

        ####################################
        # Convex MS, glPenalty = 3, L2
        ####################################
        tune.grid2 <- as.data.frame(  expand.grid( c(lambda), ncol(full) - 2) ) 
        colnames(tune.grid2) <- c("lambda", "rho")
        tune.grid2 <- unique(tune.grid2)

        # tune multi-study with l0 penalty with GL Penalty = TRUE
        tuneMS <- sparseCV_iht_par(data = full,
                                   tune.grid = tune.grid2,
                                   hoso = MSTn, # could balancedCV (study balanced CV necessary if K =2)
                                   method = "MS", # could be L0 for sparse regression or MS # for multi study
                                   nfolds = nfold,
                                   cvFolds = 5,
                                   juliaPath = juliaPath,
                                   juliaFnPath = juliaFnPath,
                                   messageInd = FALSE,
                                   LSitr = NA, # convex
                                   LSspc = NA,
                                   threads = tuneThreads)
        
        MSparams <- tuneMS$best # parameters
        
        betasMS = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                        y = as.vector(full$Y),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = b,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 0) # convex
 
        resMat[iterNum, 101] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 179] <- MSparams$rho # cardinality of support
        resMat[iterNum, 230] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm( MSparams, tuneMS)
        #######################################################
        print(paste("iteration: ", iterNum, " Complete!"))
        print(resMat[iterNum,])
        
        # time difference
        timeEnd <- Sys.time()
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 140] <- as.numeric(difftime(timeEnd, timeStart, units='mins'))
        
        timeEndTotal <- Sys.time()
        resMat[iterNum, 234] <- as.numeric(difftime(timeEndTotal, timeStartTotal, units='mins'))

        #####################################################################
        ##############
        # OSE L0Learn
        ##############
        print(paste("iteration: ", iterNum, " L0Learn OSE"))
        betaMat <- matrix(NA, ncol = K, nrow = ncol(full) - 1) # matrix of betas from each study
        L0_tune <- matrix(NA, nrow = K, ncol = ncol(tune.grid) ) # save best parameter values
        L0_tune <- as.data.frame(L0_tune)
        colnames(L0_tune) <- colnames(tune.grid)
        
        for(j in 1:K){
            
            indx <- which(full$Study == j) # rows of each study
            gm <- tune.grid$lambda / 2 # convert into comparable numbers for L0Learn
            
            # fit l0 model on jth study
            cvfit = L0Learn.cvfit(x = as.matrix(full[indx, -c(1,2)]),
                                  y = as.vector(full$Y[indx]),
                                  seed = 1,
                                  penalty="L0L2",
                                  nGamma = length(gm),
                                  algorithm = "CD",
                                  maxSuppSize = max(tune.grid$rho))
            
            # optimal tuning parameters
            optimalGammaIndex <- which.min( lapply(cvfit$cvMeans, min) ) # index of the optimal gamma identified previously
            optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
            optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
            L0LearnCoef <- coef(cvfit, lambda=optimalLambda, gamma = cvfit$fit$gamma[optimalGammaIndex] )
            
            # save tuned parameter values
            rhoStar <- sum(  as.vector(L0LearnCoef)[-1] != 0   ) # cardinality
            L0_tune$rho[j] <- rhoStar
            
            # use L0Learn coefficients as warm starts
            betaMat[,j] <- as.vector(L0LearnCoef) # save coefficients -- use "betas" as warm start for later
            rm(cvfit, indx)
        }
        
        resMat[iterNum, 165] <- mean(L0_tune$rho) # cardinality of support
        resMat[iterNum, 222] <- multiTaskRmse(data = mtTest, beta = betaMat)
        
        rm(betaMat)
        ########################
        # save results
        ########################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        
        #####################################################################
        # format data for multi-task learning packages
        Xlist <- list(length = K)
        Ylist <- list(length = K)
        groupIDs <- list(length = K)
        
        for(kk in 1:K){
          idx <- which(full$Study == kk)
          Xlist[[kk]] <- cbind(1, full[idx, Xindx]) # same design matrix for all
          Ylist[[kk]] <- full$Y[idx]
          groupIDs[[kk]] <- full$Study[idx]
          full <- full[-idx,]
        }
        #######################
        # sparse group lasso
        #######################
        print(paste("iteration: ", iterNum, " sGL"))
        
        # convert into form used by `sparsegl` package
        Xlist <- lapply(Xlist, function(x) as.matrix(x[,-1]) ) # make matrix and remove column of 1s for intercept
        Xlist <- Matrix::bdiag(Xlist)
        Ymeans <- unlist(lapply(Ylist, mean)) #colMeans(full[,Yindx]) # means for intercept
        Ylist <- lapply(Ylist, function(x) scale(x, center = TRUE, scale = FALSE)) # center Ys for each task
        Ylist <- as.numeric(do.call(c, Ylist))
        groupIDs <- rep(1:numCovs, K)
        
        # hyperparameter tuning
        asprse_mat <- data.frame(matrix(NA, nrow = 10, ncol = 3))
        colnames(asprse_mat) <- c("rmse", "lambda", "alpha")
        asprse_mat$alpha <- seq(0,1, length = nrow(asprse_mat)) # 2nd hyperparameter
        asprse_mat2 <- asprse_mat
        beta_matrix <- matrix(NA, ncol = nrow(asprse_mat), nrow = ncol(Xlist) + 1) # store best models
        
        for(asprs in 1:nrow(asprse_mat)){
          print(asprs)
          
          # tune model
          tuneMS <- cv.sparsegl(x = Xlist,
                                y = Ylist,
                                group = groupIDs,
                                intercept = FALSE,
                                nfolds = nfold,
                                nlambda = tune_length*50, # ensure enough values at the specific rho value
                                asparse = asprse_mat$alpha[asprs])
          
          sprs_const <- sapply(tuneMS$lambda, function(x) sum(coef(tuneMS, s = x) != 0 ) / K) # which coefs to be considered by sparsity
          sprs_const <- which(sprs_const <= max(rho))
          min_idx <- which.min(tuneMS$cvm[sprs_const])
          asprse_mat$rmse[asprs] <- min(tuneMS$cvm[sprs_const])
          asprse_mat$lambda[asprs] <- (tuneMS$lambda[sprs_const])[min_idx] # save results
          beta_matrix[,asprs] <- as.numeric(coef(tuneMS, s = asprse_mat$lambda[asprs] ))
          
          # no constraints 
          asprse_mat2$rmse[asprs] <- min(tuneMS$cvm)
          asprse_mat2$lambda[asprs] <- tuneMS$lambda.min # save results
          rm(tuneMS)
        }
        
        # tuned hyperparameters
        asparse <- asprse_mat$alpha[which.min(asprse_mat$rmse)]
        lambdaMin <- asprse_mat$lambda[which.min(asprse_mat$rmse)]
        beta_sprgl <- beta_matrix[,which.min(asprse_mat$rmse)] #as.numeric(coef(MSparams))
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        # results
        curIdx <- 235
        
        resMat[iterNum, curIdx+1] <- lambdaMin # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+4] <- multiTaskRmse(data = mtTest, beta = betasMS)
        resMat[iterNum, curIdx + 9] <- asparse # alpha
        curIdx <- curIdx + 9 # update
        rm( beta_sprgl, lambdaMin, idxMatrix)
        ########################################################
        # sparse group lasso
        # no sparsity constraint
        
        # tuned hyperparameters
        asparse <- asprse_mat2$alpha[which.min(asprse_mat2$rmse)]
        lambdaMin <- asprse_mat2$lambda[which.min(asprse_mat2$rmse)]
        
        MSparams <-  sparsegl(x = Xlist,
                              y = Ylist,
                              family = "gaussian",
                              group = groupIDs,
                              intercept = FALSE,
                              asparse = asparse,
                              lambda = lambdaMin)
        
        beta_sprgl <- as.numeric(coef(MSparams))
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        # results
        resMat[iterNum, curIdx+1] <- MSparams$lambda # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+4] <- multiTaskRmse(data = mtTest, beta = betasMS)
        resMat[iterNum, curIdx + 9] <- asparse # alpha
        curIdx <- curIdx + 9 # update
        
        rm(betasMS, beta_sprgl, lambdaMin, idxMatrix, MSparams)        
        ########################
        # group lasso
        ########################
        print(paste("iteration: ", iterNum, " GL not-sparse"))
        
        # hyperparameter tuning
        tuneMS <- cv.sparsegl(x = Xlist,
                              y = Ylist,
                              group = groupIDs,
                              nfolds = nfold,
                              intercept = FALSE,
                              nlambda = tune_length*50, # ensure enough values at the specific rho value
                              asparse = 0)
        
        sprs_const <- sapply(tuneMS$lambda, function(x) sum(coef(tuneMS, s = x) != 0 ) / K) # which coefs to be considered by sparsity
        sprs_const <- which(sprs_const <= max(rho))
        min_idx <- which.min(tuneMS$cvm[sprs_const])
        lambdaMin <- (tuneMS$lambda[sprs_const])[min_idx] # save results 
        beta_sprgl <- as.numeric(coef(tuneMS, s = lambdaMin)) 
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        resMat[iterNum, curIdx+1] <- lambdaMin #MSparams$lambda # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+4] <- multiTaskRmse(data = mtTest, beta = betasMS)
        resMat[iterNum, curIdx + 9] <- 0
        curIdx <- curIdx + 9 # update
        
        rm(betasMS, beta_sprgl, lambdaMin, idxMatrix)
        ########################################################
        # group lasso
        # no sparsity constraint
        MSparams <-  sparsegl(x = Xlist,
                              y = Ylist,
                              family = "gaussian",
                              group = groupIDs,
                              intercept = FALSE,
                              asparse = 0,
                              lambda = tuneMS$lambda.min)
        
        beta_sprgl <- as.numeric(coef(MSparams))
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        resMat[iterNum, curIdx+1] <- MSparams$lambda # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+4] <- multiTaskRmse(data = mtTest, beta = betasMS)
        resMat[iterNum, curIdx + 9] <- 0
        curIdx <- curIdx + 9 # update
        
        rm(MSparams, beta_sprgl, idxMatrix, tuneMS)
        ########################
        # grpreg
        ########################
        print(paste("iteration: ", iterNum, " grpreg"))
        
        grp_penalties <- c("grMCP",  "gel", "cMCP", "grLasso")
        Xlist <- as.matrix(Xlist)
        for(grp_penalty in grp_penalties){
          print(grp_penalty)
          # hyperparameter tuning
          asprse_mat <- data.frame(matrix(NA, nrow = 1, ncol = 2)) 
          colnames(asprse_mat) <- c("rmse", "lambda")
          beta_vec <- vector(length = ncol(Xlist) + 1) # store best models
          
          # tune model
          # hyperparameter tuning
          tuneMS <- cv.grpreg(X = Xlist,
                              y = Ylist,
                              # fold = foldID,
                              penalty = grp_penalty, 
                              family = "gaussian",
                              group = groupIDs,
                              intercept = FALSE,
                              nlambda = tune_length*50, # ensure enough values at the specific rho value
                              seed = 1)
          
          sprs_const <- sapply(1:length(tuneMS$lambda), function(x) sum(tuneMS$fit$beta[-1,x] != 0 ) / K) # which coefs to be considered by sparsity
          sprs_const_rho <- which(sprs_const <= rho)
          print(length(sprs_const_rho))
          # make sure there are some of the correct length
          if(length(sprs_const_rho) == 0){
            # if none, use the rho that is largest one that still is within constraint
            rho_max <- max(sprs_const[sprs_const <= rho]) # biggest rho 
            print(paste("rhomax",rho_max))
            sprs_const <- which(sprs_const == rho_max)
          }else{
            sprs_const <- sprs_const_rho
          }
          
          if(length(sprs_const) >= tune_length){
            sprs_const <- sample(sprs_const, size = tune_length) # sample to ensure equal numbers of tuning parameters 
          }          
          
          min_idx <- which.min(tuneMS$cve[sprs_const])
          asprse_mat$rmse <- min(tuneMS$cve[sprs_const])
          asprse_mat$lambda <- (tuneMS$lambda[sprs_const])[min_idx] # save results
          beta_vec <- tuneMS$fit$beta[, sprs_const[min_idx] ]
          rm(tuneMS)
          
          # tuned hyperparameters
          lambdaMin <- asprse_mat$lambda
          beta_sprgl <- beta_vec[-1] # remove intercept since it will be approx 0
          
          # make into matrix
          idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
          beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
          betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on

          resMat[iterNum, curIdx+1] <- lambdaMin # tuning parameter
          resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
          resMat[iterNum, curIdx+4] <- multiTaskRmse(data = mtTest, beta = betasMS)
          resMat[iterNum, curIdx + 9] <- asparse # alpha
          curIdx <- curIdx + 9 # update
          rm(betasMS, beta_sprgl, lambdaMin, idxMatrix)
        }

        ########################################################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
