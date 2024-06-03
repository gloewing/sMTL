# Updates: December 19, 2021 
# uses active set versions appropriate for each method
# list.of.packages <- c("pROC", "JuliaConnectoR", "caret", "glmnet",  'L0Learn','sparsegl',
#                       "MASS",  "Matrix", "grpreg", "RMTL", "mixtools") 
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
# if(length(new.packages)){
#   chooseCRANmirror(ind=75)
#   install.packages(new.packages, dependencies = TRUE)
# } 
library(pROC)
library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)
library(sparsegl)
library(grpreg)
library(MASS)
library(RMTL)
library(Matrix)
library(mixtools)
#######################################

# sims <- expand.grid(1, 50, 1e-10, c(50,100), 5, 1, "exponential", c(7,10,13), 0.2, 0.5, 250, 0, 20, 0.5, 2, "multiTask", TRUE, c(0, seq(10,20, by = 2),20, 30, 40) )
# colnames(sims) <- c("simNum", "betaVar", "xVar", "nLow",
#                         "K", "tau", "cov", "rho",
#                          "betaRangeLow", "betaRangeHigh", "p", "s", "r", "r_p", "errorMult", "multiTask", "tuneInd", "q")
# sims <- sims[order(sims$nLow),]

# # # s is true number of non-zeros
# # r is number of covariates that are potentially divergent support (set of covariates that are randomly selected to be include)
# # r_p is the probability of inclusion for a given study (i..e, each of the K studies selects each of the r covariates for inclusion ~ i.i.d. Ber(r_p) )
# write.csv(sims, "~/Desktop/Research/sparseParam_test", row.names = FALSE)

cluserInd <- TRUE # whether running on computer or on cluster
   
if(cluserInd){
    
  # this R code is saved in: /home/loewingergc/smtl/code
  # bash in: /home/loewingergc/bash
  # parameters sims6 in: /home/loewingergc/smtl/data
  
  # paths
  wd <- simWD <- "/home/loewingergc/smtl/code/" 
  data_path <-"/home/loewingergc/smtl/data" # path to original Matlab files for data
  save.folder <- "/home/loewingergc/smtl_sims0" # simulation_scienc
  
  #################
  # if on cluster
  args = commandArgs(TRUE)
  runNum <- as.integer( as.numeric(args[1]) ) 
  iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id
  
  juliaPath <- "/usr/local/apps/julialang/1.7.3/bin" #"/usr/local/lmod/modulefiles"
  juliaFnPath_MT <- juliaFnPath <- "/home/loewingergc/smtl/code/"
  sims6 <- read.csv(paste0(data_path,"/sparseParam_test"))
  
}else{
  runNum <- 1
  setwd("~/Desktop/Research")
  sims6 <- read.csv("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Code/sparseParam_test")
  iterNum <- 1
  wd <- "/Users/loewingergc/Downloads/sMTL-main/sMTL_Paper/sMTL_Functions/"
  simWD <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 5 Sims/Resubmission 6-29-23/"
  # Julia paths
  juliaPath = "/Applications/Julia-1.7.app/Contents/Resources/julia/bin"
  juliaFnPath_MT <- juliaFnPath <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/IHT/Tune MT/"
  save.folder <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study"
}

# try with tau = 10, consider playing with correlation of features
# K = 5
######################################
source(paste0(simWD, "SimFn.R"))

Sys.setenv(JULIA_BINDIR = juliaPath)

# sim params
simNum <- runNum
totalSims <- 100

# simulation parameters
totalStudies <- sims6$K[runNum] # need number that is divisible by 2, 4, 8 for cluster purposes and structure of
clusts <-  1:totalStudies #sims6[runNum, 4] # totalStudies
betaVar <- sims6$betaVar[runNum]
covVar <- sims6$xVar[runNum]
trainStudies <- 1:(totalStudies - 1)
K <- length(trainStudies)
num.trainStudy <- length(trainStudies)
totalTestSz <- 150 # number including training test set and actual test set
testTrain <- 50
tune_length <- 100 # number of parameters to tune over to keep constant across methods
rho <- sims6$rho[runNum] # NEED TO UPDATE SIMULATION GRID
 
categor <- 4 # if true: cardinality of random support is fixed, if FALSE: random (iid Bernouli) 
p <- 0
# q <- 5
numCovs <- sims6$p[runNum] #p + q + 45 # p and q here refer to non-zero coefficients
s <- sims6$s[runNum] #
r <- sims6$r[runNum]
r_p <- sims6$r_p[runNum]
q <- sims6$q[runNum] # number of indices to select r_card from for categ = 4 (if q is large, less common support on average), q * 2 + 2* s must be <= p
zeroCovs <- seq(2, numCovs + 1)
if(s > 0)   zeroCovs <- seq(2, numCovs + 1)[-seq(2, 2 * s, by = 2)] # alternate because of exponential correlation structure
# Study Strap and Model fitting parameters
test_study <- max(trainStudies) + 1 # arbitrarily set to study but any of the non-training studies is fine (11-24 are all random test studies)
scaleInd <- TRUE
betaMeanRange <- c(sims6$betaRangeLow[runNum],   sims6$betaRangeHigh[runNum])
clustDiv <- 10
Mmultiplier <- 1.5
timeLimTn <- 60 # time limit for tuning
timeLim <- 1200 # time limit for running
seedFixedInd <- TRUE # fixed effects (true betas) and Sigma_x fixed across simulation iterations
rho_corr <- 0.5 # rho used in covariance matrix for features if "exponential" or "pairwaise"
covType <- sims6$cov[runNum] # type of covariance matrix for the features
MoM <- FALSE # indicator of whether to run MoM
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- 50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- 1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 50 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
maxIter_cv <- 5000 # originally 5000, was fast with 1000
mix_prob <- NA # 3/4 # proportion of mixture of gaussians that has mean 0 -- NA is not a mixture
mix_sd <- NA #5 # multiplier of standard dev of random effects that is greater than 0 -- NA is not a mixture

errorMult <- sims6$errorMult[runNum] # range of error for uniform
tau <- sims6$tau[runNum]
epsHigh <- tau * errorMult# noise lower/upper
epsLow <- tau / errorMult# noise lower/upper
nLow <- nHigh <- sims6$nLow[runNum]  # multiply by 2 because of multi-task test set  # samp size lower/upper
tuneInd <- sims6$tuneInd[runNum]
# nHigh <- sims6$nLow[runNum] * errorMult # multiply by 2 because of multi-task test set  # samp size lower/upper
# nLow <- sims6$nLow[runNum] / errorMult
WSmethod = 1 # sims6$WSmethod[runNum] ---- 2
ASpass = TRUE # sims6$ASpass[runNum] ----- TRUE # next

if(tuneThreads == 1){
    # use non-parallel version
  source(paste0(wd, "sparseFn_iht_test_MT.R")) # USE TEST VERSION HERE
  sparseCV_iht_par <- sparseCV_iht
}else{
    # source("sparseFn_iht_par.R")
  source(paste0(wd, "sparseFn_iht_test_MT.R")) # USE TEST VERSION HERE
  sparseCV_iht_par <- sparseCV_iht
}

# model tuning parameters
L0TuneInd <- TRUE # whether to retune lambda and rho with gurobi OSE (if FALSE then use L0Learn parameters)
L0MrgTuneInd <- TRUE # whether to retune lambda and rho with gurobi Mrg (if FALSE then use L0Learn parameters)
L0_sseTn <- "sse" # tuning for L0 OSE (with gurobi not L0Learn)
MSTn <- sims6$multiTask[runNum] #"hoso" #"balancedCV" # tuning for MS (could be "hoso")
nfold <- 10

lamb <- 0.5
fileNm <- paste0("MTL",
                "_s_", s, "_r_", r, "_rp_", r_p,
                "_q_", q,
                  "_p_", numCovs, "_n_", nLow, ".", nHigh,
                 "_eps_", epsLow, ".", epsHigh,
                "_covTyp_", covType,
                "_rho_", rho_corr,
                "_cDv_", clustDiv,
                "_bVar_", betaVar, "_xVar_",
                round( covVar, 2), 
                "_K_", K,
                "_bMu_", betaMeanRange[1], "_", betaMeanRange[2], "_",
                "_bFx_", seedFixedInd,
                "_sseTn_", L0_sseTn,
                "_MSTn_", MSTn,
                "_nFld_", nfold,
                "_LSitr_", LSitr,
                "_LSspc_", LSspc,
               "_wsM_", WSmethod,
               "_asP_", ASpass,
               "_TnIn_", tuneInd,
               "cat_", categor,
               "_mxP_", mix_prob,
               "_mxSD_", mix_sd,
               "_rho_", rho)

timeStart <- Sys.time()

##############
#######################################################
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

    resMat <- matrix(nrow = totalSims, ncol = 289)
    colnames(resMat) <- c("mergeRdg", "oseRdg","oseRdgStk",
                          "mrgL0", "mrgL0_fp", "mrgL0_tp",
                          "mrgL0_sup", "mrgL0_auc", "mrgL0Lrn",
                          "mrgL0Lrn_fp", "mrgL0Lrn_tp",
                          "mrgL0Lrn_sup", "mrgL0Lrn_auc",
                          "oseL0Lrn","oseL0LrnStk", "oseL0Lrn_fp",
                          "oseL0Lrn_tp", "oseL0Lrn_sup", "oseL0Lrn_auc",
                          "oseL0LrnStk_fp", "oseL0LrnStk_tp", "oseL0LrnStk_sup",
                          "oseL0LrnStk_auc", "studyL0Lrn_aucAvg", "oseL0",
                          "oseL0_L2_lambda", "oseL0_fp", "oseL0_tp", "oseL0_sup",
                          "oseL0_auc", "oseL0Stk_fp", "oseL0Stk_tp", "oseL0Stk_sup",
                          "oseL0Stk_auc", "studyL0_aucAvg",
                          "msP1_L0", "msP1_L0Stk", "msP1_L0_fp", "msP1_L0_tp",
                          "msP1_L0_sup", "msP1_L0_auc", "msP1_L0Stk_fp",
                          "msP1_L0Stk_tp", "msP1_L0Stk_sup", "msP1_L0Stk_auc",
                          "msP2_L0","msP2_lambdaL2", "msP2_L0_fp", "msP2_L0_tp",
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
                          "MoM_mrg_auc", "MoM_mrg_rho",
                          ##
                          "MoM_mrg_L0", "MoM_mrg_L0_fp", "MoM_mrg_L0_tp",
                          "MoM_mrg_L0_sup", "MoM_mrg_L0_auc", "MoM_mrg_L0_rho",

                          "lambda_L2", "MoM_Stk", "MoM_fp", "MoM_tp",
                          "MoM_sup", "MoM_auc", "MoM_Stk_fp",
                          "MoM_Stk_tp", "MoM_Stk_sup", "MoM_Stk_auc", 
                          "suppOverlap",
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
                          "s_MS3con", "s_MS3conStk", "s_MoM", "s_MoM_L0", "s_MoM_ose", "s_MoM_Stk","s_MoML0_ose", "s_MoML0_Stk",
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
                          paste0("GL_",
                                 c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                          paste0("grMCP_",
                                 c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                          paste0("gel_",
                                 c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                          paste0("cMCP_",
                                 c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha")),
                          paste0("lasso_",
                                 c("lambda", "s", "coef", "MT", "fp", "tp", "sup", "auc", "alpha"))
                          )

        
        minRho <- max(   c(1, (s - 2) )   )
        maxRho <- min(   numCovs,  (s+r+1)  )
        
        # save results
        studyNum <- test_study
        cnt <- seedSet <- iterNum # ensures no repeats
        set.seed(seedSet)

        # if true, then we fix the fixed effects (true betas) across iterations of simulation and just allow random effects to vary
        # also fixes Sigma
        if(seedFixedInd)   seedFixed <- 1 # arbitrarily set seed to 1 so fixed across iterations at seedFixed=1

        if(nLow == nHigh){
            # if the samople sizes are all the same
            sampSizeVec <- c(   rep(nLow, totalStudies - 1),
                                totalTestSz )

        }else{
            sampSizeVec <- c(sample(nLow:nHigh, totalStudies - 1, replace = TRUE), totalTestSz)
        }

        ################################
        # generate inclusion variables for support
        ################################
        fixB <- matrix(1, nrow = totalStudies, ncol = numCovs + 1)

        suppSeq <- seq(2*s + 3, 2*(s + r) + 1, by = 2) # alternating sequence of covariate indices starting after common support that is "r" long
        
        for(j in 1:totalStudies){
            
            if(categor == TRUE){
                # fixed cardinality: categorical random variables
                r_card <- round(r * r_p) # cardinality: total number of 1s is roughly same in expectation
                suppRandom <- c( rep(1, r_card), rep(0, r - r_card) )
                suppRandom <- suppRandom[ sample.int(r, replace = FALSE) ] # shuffle order
            
                # cardinality of support
                card <- r_card + s # all have the same cardinaity so choose first task arbitrarily
                
                # 3 above and below true support
                minRho <- max(1,   card - 3   )
                maxRho <- min(   card + 3  )
                
            }else if(categor == FALSE){
                # random cardinality, bernoulli random variables
                suppRandom <- rbinom(r, 1, prob = r_p) # iid bernoulli draws that is r long
                
            }else if(categor == 2){
                # complete disjoint sets
                suppSeq <- seq(2*s + 3, numCovs, by = 2) # rest of variables
                r_card <- round(r * r_p) # cardinality: total number of 1s is roughly same in expectation
                suppRandom <- seq( (j - 1) * r_card + 1, j * r_card)
                
                # cardinality of support
                card <- r_card + s # all have the same cardinaity so choose first task arbitrarily
                
                # 3 above and below true support
                minRho <- max(1,   card - 3   )
                maxRho <- min(   card + 3  )
            }else{
                # throw away
                suppRandom <- rbinom(r, 1, prob = r_p) # iid bernoulli draws that is r long
                
            }
            
            suppIndx <- suppSeq * suppRandom # the indices multiplied by the bernoulli draw: then we keep the indices that are non-zero
            retainIndx <- which(  zeroCovs %in% suppIndx[suppIndx > 0] ) # find indices that were selected above

            # remove the indices that were selected
            if(length(retainIndx) > 0){
                zeroIndx <- zeroCovs[-retainIndx]
            }else{
                zeroIndx <- zeroCovs
            }

            fixB[j, zeroIndx] <- 0 # only add the ones that are not zeroed out
        }
        

        ##########################################
        # exact cardinality different way to construct support
        if(categor == 4){
            fixB <- matrix(0, nrow = totalStudies, ncol = numCovs + 1)
            
            # common support
            if(s > 0){
                sSeq <- seq(1, 2 * s, by = 2) 
                fixB[, sSeq ] <- 1 # only add the ones that are not zeroed out
            }else{
                sSeq <- 0
            }
           
            # cardinality
            r_card <- round(r * r_p) # cardinality: total number of 1s is roughly same in expectation
            
            if(q > 0){
              # means there can be SOME overlap in supports
              q <- max(r_card, q) # make sure it is big enough to select at least r_card
              
              m <- max(sSeq) + 2
              suppSeq <- seq(m, m + 2 * (q - 1), by = 2) #seq(2*s + 3, 2*(s + r) + 1, by = 2) # alternating sequence of covariate indices starting after common support that is "r" long
            }

            # draw supports
            if(q == 0){
              # this is a way to set the supports to have ZERO overlap. For all other integers, q > 0: larger q means more heterogeneity (not less)
              supp_start <- 2
              for(j in 1:totalStudies){
                # fixed cardinality: categorical random variables
                suppRandom <- seq(supp_start, supp_start + 2*(r_card-1), by = 2)
                fixB[j, suppRandom ] <- 1 # only add the ones that are not zeroed out
                supp_start <- supp_start + 2 * r_card # update support so there is no overlap with previous study
              }
            }else{
              # q > 0 means there is (potentially) some support overlap
              for(j in 1:totalStudies){
                # fixed cardinality: categorical random variables
                suppRandom <- sample(suppSeq, r_card, replace = FALSE)
                fixB[j, suppRandom ] <- 1 # only add the ones that are not zeroed out
              }
            }

            
            # cardinality of support
            card <- sum( fixB[1,] ) # all have the same cardinaity so choose first task arbitrarily
            
            # 3 above and below true support
            minRho <- max(1,   card - 3   )
            maxRho <- min(   card + 3  )
            
        }
        # Z <- matrix(0, nrow = p, ncol = K)
        
        # overlap of true support
        resMat[iterNum, 129] <- suppressWarnings( suppHet(t(fixB), intercept = FALSE)[1] )
        
        
        print(fileNm)

        ##########################################
        #rho <- minRho:maxRho
        lambda <- sort( unique( c(exp(seq(0, 6, length = tune_length/2)),
                                  exp(-seq(0, 25, length = tune_length/2))
                                  ) ), decreasing = TRUE ) 
        
        lambdaZ <- sort( unique( c(0, 
                                   exp(seq(0,4.75, length = round(tune_length/2) - 11)),
                                   exp(-seq(0,25, length = round(tune_length/2) - 11))) ),
                         decreasing = TRUE ) 
        
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
        
        tune.grid <- as.data.frame(  expand.grid(
            c(lambda) , # 0 # add 0 but not to glmnet because that will cause problems
            rho)
        ) # tuning parameters to consider
        colnames(tune.grid) <- c("lambda", "rho")
        
        # glmnet for ridge
        tune.gridGLM <- as.data.frame( cbind( 0, lambda) ) # Ridge
        
        colnames(tune.gridGLM) <- c("alpha", "lambda")
        
        ##########################################
        rfB <- fixB
        

        # simulate data
        full <- multiStudySimNew_MT(
                               seed = seedSet, # general seed for simulations
                               seedFixed = seedFixed,
                               sampSize = sampSizeVec * 2, # multiple by 2 since we split into 2 for multi-task learning belo
                               num.studies = totalStudies,
                               covariate.var = covVar, # scales the variance of the MVNormal that generates the true means of the covaraites
                               beta.var = betaVar, # variance of random effects
                               cluster.beta.var = betaVar / clustDiv, # variance of cluster specific random effect
                               cluster.X.var = betaVar / clustDiv, # variance of cluster specific random effect
                               clusters_mu = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                               clusters_beta = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                               num.covariates = numCovs,
                               # zero_covs = zeroCovs, # indices of covariates which are 0
                               fixed.effects = NULL, # all random effects if NA #0:p, # indices of fixed effects -- 1 is the intercept
                               fixed.effectsX = c(), # indices of "fixed covariates" across studies
                               fixB = fixB, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of betas marginally)
                               rfB = rfB, # study specific boolean (if TRUE then there is study specific random effects for betas)
                               cB = FALSE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for betas)
                               fixX = TRUE, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of covariates marginally)
                               rfX = FALSE, # study specific boolean (if TRUE then there is study specific random effects for X)
                               cX = FALSE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for X)
                               studyNoise = c(epsLow, epsHigh), # range of noises for the different studies
                               beta.mean.range = betaMeanRange, # true means of hyperdistribution of beta are drawn from a unif(-beta.mean.range, beta.mean.range)
                               params = TRUE,
                               sigmaDiag = FALSE, # if true then the covariance matrix of covariates (X and Z) is diagonal
                               sigmaIdentity = FALSE, # if true then covariance matrix is the identity matrix
                               Xmeans_constant = FALSE, # if TRUE then the means of all covariates in a study are shifted by the same amount
                               XZmeans_constant = FALSE, # if TRUE then the means of all the X (fixed effects) covariates and Z (random effect) covariates each separately have all their means the same (all Xs same and all Zs same but X and Z different)
                               Xmean0 = TRUE, # IF TRUE then the marginal distribution of the Xs is mean 0
                               covariance = covType, # "random"- qr decomposition, "identity", "exponential rho", "pariwise rho"
                               corr_rho = rho_corr,#, # used if pariwise or exponential correlation
                               mix_probs = c(mix_prob, 1 - mix_prob), # mixture probabilities: if NA then no mixture of gaussians
                               mix_mu = c(0, betaVar * mix_sd) # mixture means of mix of gaussians: if NA then no mixture of gaussians -- 2 standard deviations of random effects above 0
                               )

        ### SNR
        #-----------------------------
        trueZ <- I(full$betas != 0) * 1
        trueB <- t( full$betas )# [,-test_study] # true betas
        #-----------------------------

        full <- as.data.frame( full$data )
        
        # split multi-study
        mtTest <- multiTaskSplit_MT(data = full, split = 0.5) # make training sets
        full <- mtTest$train
        mtTest <- mtTest$test
        # vector of true indicators of whether 0 or not
        
        z <- rep(1, numCovs)
        z[zeroCovs - 1] <- 0

        K <- totalStudies # length(countries) # number of training countries
        Yindx <- 1:K
        Xindx <- seq(K + 1, ncol(full) )
        
        ####################################
        # scale covariates
        ####################################

        if(scaleInd == TRUE){

            nFull <- nrow( full ) # sample size of merged

            # scale Covaraites
            means <- colMeans( as.matrix( full  ) )
            sds <- sqrt( apply( as.matrix( full  ), 2, var) *  (nFull - 1) / nFull )  # use mle formula to match with GLMNET

            # columns 1 and 2 are Study and Y
            for(column in (K+1):ncol(full) ){
                # center scale
                full[, column] <- (full[, column ] - means[column]) / sds[column]
                mtTest[, column] <- (mtTest[, column ] - means[column ]) / sds[column]
                
            }


        }else if(scaleInd == "spec"){
            # scale according to rows of training set of test country


            nFull <- nrow(X[trainIndx,]) # sample size of merged

            # scale Covaraites
            means <- colMeans( as.matrix(X[trainIndx,]) )
            sds <- sqrt( apply( as.matrix(X[trainIndx,]), 2, var) *  (nFull - 1) / nFull )  # use mle formula to match with GLMNET

            #
            for(column in 1:ncol(X) ){
                # center scale
                X[, column] <- (X[, column ] - means[column]) / sds[column]
                test2[, column + 2] <- (test2[, column + 2] - means[column]) / sds[column]
            }

            # update the design matrix
            full[,Xindx] <- X

        }

        mtTest <- mtTest  # remove study labels from multi-task test set
        
        rownames(mtTest) <- 1:nrow(mtTest)
        rownames(full) <- 1:nrow(full)

        timeStartTotal <- Sys.time()
        #######################
        # Tuning for Ridge GLMNET
        #######################
        # # Merge Tune for Ridge
        print(paste("iteration: ", iterNum, " Tune Ridge Meg/OSE"))

        ##############
        # GLMNET Ridge - MultiTask
        ##############
        allRows <- 1:nrow(full)
        set.seed(1)
        HOOL <- createFolds(allRows, k = nfold) # make folds 
        foldID <- vector( length = nrow(full) )
        for(fd in 1:nfold)  foldID[ HOOL[[fd]] ] <- fd

        tune.mod <- cv.glmnet(y = as.matrix(full[,Yindx]),
                              x = as.matrix(full[,Xindx]),
                              alpha = 0,# tune.gridGLM$alpha,
                              # lambda = tune.gridGLM$lambda,
                              intercept = TRUE,
                              foldid = foldID,
                              family = "mgaussian")

        lambdaStar <- tune.mod$lambda.min


        betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = "lambda.min") ) )
        betaEst <- as.matrix( betaEst )
        
        resMat[iterNum, 203] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 217] <- multiTaskRmse_MT(data = mtTest, beta = betaEst)
        rm( betaEst, tune.mod )

        ##############
        # GLMNET Lasso - MultiTask -- DOES NOT CONSTRAIN SUPPORT SIZE
        ##############
        print(paste0("MLT Lasso glmnet: ", iterNum))
        tune.mod <- cv.glmnet(y = as.matrix(full[,Yindx]),
                              x = as.matrix(full[,Xindx]),
                              alpha = 1,# tune.gridGLM$alpha,
                              nlambda = tune_length,
                              foldid = foldID,
                              intercept = TRUE,
                              family = "mgaussian")
        
        lambdaStar <- tune.mod$lambda.min
        
        
        betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = "lambda.min") ) )
        betaEst <- as.matrix( betaEst )

        resMat[iterNum, 203] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 218] <- multiTaskRmse_MT(data = mtTest, beta = betaEst)

        zAvg <- rowMeans( I( betaEst[-1,] != 0) * 1 ) # stacking
        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat( response = trueZ[j, -1], predictor = zAvg )
        }

        resMat[iterNum, 103:106] <- colMeans(supMat) # suppStat(response = z, predictor = lassoSupp[-1])
        rm(betaEst, tune.mod, supMat)
        
        resMat[iterNum, 162] <- sum(zAvg) # size of xupport

        ##############
        # OSE Ridge
        ##############
        timeStart2 <- Sys.time()
        print(paste("iteration: ", iterNum, " Ridge OSE"))
        b <- matrix(0, ncol = K, nrow = numCovs + 1)
        
        tune.grid_sse <- data.frame(lambda1 = unique(lambda),
                                    lambda2 = 0,
                                    lambda_z = 0,
                                    rho = numCovs)
        
        tune.grid_sse <- unique( tune.grid_sse[tune.grid_sse$lambda1 > 0,] )
        
        L0_tune <- sparseCV_iht_par(data = full,
                                    tune.grid = tune.grid_sse,
                                    hoso = MSTn, #L0_sseTn,  # could balancedCV (study balanced CV necessary if K =2)
                                    method = "MS_z_fast", # this does not use the active set so should be much faster and since we do not use a lambda_z penalty it is OK to not use active set
                                    nfolds = nfold,
                                    cvFolds = 5,
                                    juliaPath = juliaPath,
                                    juliaFnPath = juliaFnPath,
                                    messageInd = FALSE,
                                    LSitr = NA,
                                    LSspc = NA,
                                    maxIter = maxIter_cv,
                                    threads = tuneThreads
        )
        
        L0_tune <- L0_tune$best
        
        # final model
        betasMS = L0_MS_z2(X = as.matrix( full[ , Xindx ]) ,
                           y = as.matrix( full[, Yindx] ),
                           rho = numCovs,
                           study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                           beta = b,
                           lambda1 = L0_tune$lambda1,
                           lambda2 = 0,
                           lambda_z = 0,
                           scale = TRUE,
                           maxIter = 10000,
                           localIter = 0
        )
        
        predsMat <- cbind( 1, as.matrix( full[,Xindx ] ) ) %*% betasMS
        
        resMat[iterNum, 205] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 219] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        timeEnd2 <- Sys.time()
        resMat[iterNum, 235] <- as.numeric(difftime(timeEnd2, timeStart2, units='mins'))
        
        rm(predsMat)
        ###############################################################

        ##############
        # OSE L0Learn
        ##############
        print(paste("iteration: ", iterNum, " L0Learn OSE"))
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        betaMat <- betas <- matrix(NA, ncol = K, nrow = length(Xindx) + 1) # matrix of betas from each study
        res <- vector(length = K) # store auc
        Mvec <- vector(length = K) # use to initialize other models later on
        # save best parameter values
        L0_tune <- matrix(NA, nrow = K, ncol = ncol(tune.grid) ) # save best parameter values
        L0_tune <- as.data.frame(L0_tune)
        colnames(L0_tune) <- colnames(tune.grid)

        supMat <- matrix(NA, nrow = K, ncol = 4)

        for(j in 1:K){

            sdY <- 1 # set to 1 for now so we DO NOT adjust as glmnet() #sd(full$Y[indx]) * (n_k - 1) / n_k #MLE
            gm <- tune.grid$lambda / (sdY * 2) # convert into comparable numbers for L0Learn

            # fit l0 model on jth study
            cvfit = L0Learn.cvfit(x = as.matrix(full[, Xindx]),
                                  y = as.vector(full[,j]),
                                  seed = 1,
                                  penalty="L0L2",
                                  nGamma = length(gm),
                                  algorithm = "CDPSI",
                                  maxSuppSize = max(tune.grid$rho), # largest that we search
                                  scaleDownFactor = 0.99
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

            # model specific nonzero covs
            zOSE <- I(L0LearnCoef != 0) * 1 # nonzero betas

            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zOSE[-1])
            rm(cvfit)
        }


        zAvg <- I( rowMeans(betaMat)[-1] != 0) * 1 # stacking

        resMat[iterNum, 16:19] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 165] <- mean(L0_tune$rho) # cardinality of support
        resMat[iterNum, 208] <- sqrt( mean( (betaMat - trueB)^2 ) ) # coef error
        resMat[iterNum, 222] <- multiTaskRmse_MT(data = mtTest, beta = betaMat)
        
        rm(betaMat, res, supMat, zAvg, zOSE)
        ##############################################
       
        ############
        # OSE L0
        ############
        print(paste("iteration: ", iterNum, " OSE L0"))
        
        timeStart1 <- Sys.time()
        
        tune.grid_OSE <- data.frame(lambda1 = unique(lambda),
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
                                    messageInd = FALSE,
                                    LSitr = LSitr,
                                    LSspc = LSspc,
                                    threads = tuneThreads,
                                    maxIter = maxIter_cv,
                                    WSmethod = WSmethod,
                                    ASpass = ASpass
                                    )
        
        L0_tune <- L0_tune$best # parameters
        
        # initialize algorithm warm start
        p <- length(Xindx) + 1
        b <-  matrix(0, ncol = K, nrow = p ) #rep(0, p) #rnorm( p )
        
        # matrices
        betas <- matrix(NA, nrow = p, ncol = K)
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- vector(length = K) # store auc
        supMat <- matrix(NA, nrow = K, ncol = 4)
        
        # warm start
        warmStart = L0_MS_z3(X = as.matrix( full[ , Xindx ]) ,
                                     y = as.matrix( full[, Yindx] ),
                                     rho = min(L0_tune$rho * 4, length(Xindx) -1),
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
        betas = L0_MS_z3(X = as.matrix( full[ , Xindx ]) ,
                                 y = as.matrix( full[, Yindx] ),
                                 rho = L0_tune$rho,
                                 study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                                 beta = warmStart,
                                 lambda1 = L0_tune$lambda1,
                                 lambda2 = 0,
                                 lambda_z = 0,
                                 scale = TRUE,
                                 maxIter = 10000,
                                 localIter = localIters,
                                 WSmethod = WSmethod,
                                 ASpass = ASpass
                        )
        
        timeEnd1 <- Sys.time()
        
        resMat[iterNum, 233] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))

        for(j in 1:K){
            
            indx <- which(full$Study == j) # rows of each study
            
            fitG <- betas[, j]

            # stack matrix
            predsMat[,j] <- cbind( 1, as.matrix( full[,Xindx ] ) ) %*% fitG

            # model specific nonzero covs
            zOSE <- I(fitG != 0) * 1 # nonzero betas
            
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zOSE[-1])
            rm(indx)
            
        }
        rm(predsMat)

        # stacking auc
        zAvg <- I( rowMeans(betas)[-1] != 0) * 1# stacking

        resMat[iterNum, 27:30] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose

        supMat <- matrix(NA, nrow = K, ncol = 4)

        resMat[iterNum, 26] <- lambdaRidge <- L0_tune$lambda1 # ridge tuning parameter
        resMat[iterNum, 31:34] <- colMeans(supMat) #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 167] <- L0_tune$rho # cardinality of support
        resMat[iterNum, 209] <- sqrt( mean( (betas - trueB)^2 ) ) # coef error
        resMat[iterNum, 223] <- multiTaskRmse_MT(data = mtTest, beta = betas)
        
        rm(res, L0_tune, supMat, zOSE, zAvg, betas)

        ####################################
        # common support betaBar penalty
        ####################################
        # random initialization for betas
        betas <- matrix( 0, nrow = numCovs + 1, ncol = K )

        print(paste("iteration: ", iterNum, " common support Group IP"))
        glPenalty <- 1
        ip <- TRUE
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- resS <- vector(length = K) # store support prediction

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
                           maxIter = maxIter_cv,
                           threads = tuneThreads)

        MSparams <- tuneMS$best # parameters

        # warm start for 4 * rho and no betaBar regularization
        warmStart = L0_MS2(X = as.matrix( full[ , Xindx ] ),
                         y = as.matrix( full[, Yindx] ),
                         rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = betas,
                         lambda1 = 1e-6, #MSparams$lambda1,
                         lambda2 = 0, # use 0 as warm start
                         scale = TRUE,
                         maxIter = 10000,
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
                      maxIter = 10000,
                      localIter = localIters
        )


        # support stats
        # zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # avg

        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }

        resMat[iterNum, 38:41] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 42:45] <- colMeans(supMatS) # suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 96] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 169] <- MSparams$rho # cardinality of support
        resMat[iterNum, 210] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 224] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, supMat, zAvg, betasMS, warmStart)

        ########################################################

        ####################################
        #  ZBar + L2 
        ####################################
        timeStart1 <- Sys.time()
        print(paste("iteration: ", iterNum, " Zbar and Bbar"))
        
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- resS <- vector(length = K) # store support prediction
        
        if(tuneInd){
          
          # ************
          # original from 1/20/21
          # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
          # ************
          
          tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambdaRidge / 2, 0, lambdaZ, rho) )
          colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          # order correctly
          tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                     -tune.grid_MSZ_5$lambda_z,
                                                     decreasing=TRUE),     ]
          
          tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
          print(paste("iteration: ", iterNum, " Zbar and Bbar Tune 1"))
          
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
          
          rhoG <- rho 
          lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
          lambdaZgrid <- sort( unique(lambdaZgrid), decreasing = TRUE)
          
          # ************
          # original from 1/20/21
          # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
          # ************
          gridUpdate <- as.data.frame(  expand.grid( lambdaRidge / 2, 0, lambdaZgrid, rhoG) )
          colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                           gridUpdate$lambda1,
                                           -gridUpdate$lambda_z,
                                           decreasing=TRUE),     ]
          
          gridUpdate <- unique(gridUpdate)
          print(paste("iteration: ", iterNum, " Zbar and Bbar Tune 2"))
          
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
                                     ASpass = ASpass
          )
          
        }else{
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambdaRidge / 2, 0, lambdaZ, rho) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                       tune.grid_MSZ_5$lambda1,
                                                       -tune.grid_MSZ_5$lambda_z,
                                                       decreasing=TRUE),     ]
            
            tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
            
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
        print(paste("iteration: ", iterNum, " Zbar and Bbar WS"))
        
        MSparams <- tuneMS$best # parameters

        # warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
        warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                          y = as.matrix( full[, Yindx] ),
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
                         maxIter = 10000,
                         localIter = localIters,
                         WSmethod = WSmethod,
                         ASpass = ASpass
        )
        
        timeEnd1 <- timeEnd <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 231] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))

        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <-matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }

        resMat[iterNum, 48:51] <- colMeans(supMat) #suppStat(response = z,  predictor = zAvg * 1 )  # auc average weights ose
        resMat[iterNum, 97] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 47] <- MSparams$lambda1 # L2 tuning parameter
        resMat[iterNum, 171] <- MSparams$rho # cardinality of support
        resMat[iterNum, 211] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 225] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        lambdaZstar <- MSparams$lambda_z
        
        rm(MSparams, tuneMS, supMat, betasMS, zAvg, warmStart )
        ########################################################
        ####################################################
        # convex version of MS (no cardinality constraints)
        ####################################################
        print(paste("iteration: ", iterNum, " Group Convex"))
        ####################################
        # ||beta - betaBar|| and NO Frobenius norm no cardinality constraints
        # Convex, IP = FALSE
        ####################################
        tune.grid_MS2$rho <- length(Xindx) # full support
        tune.grid_MS2 <- unique(tune.grid_MS2) # convex -- no IP selection

        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        res <- vector(length = K) # store auc

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
                               LSitr = NA,
                               LSspc = NA, # convex
                               maxIter = maxIter_cv,
                               threads = tuneThreads 
        )

        MSparams <- tuneMS$best # parameters

        betasMS = L0_MS2(X = as.matrix( full[ , Xindx ]) ,
                         y = as.matrix( full[, Yindx] ),
                         rho = MSparams$rho,
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = betas,
                         lambda1 = MSparams$lambda1,
                         lambda2 = MSparams$lambda2,
                         scale = TRUE,
                         maxIter = 10000,
                         localIter = 0 # convex
        )


        # stacking auc
        zAvg <- I( rowMeans(betasMS)[-1] != 0) * 1 # stacking
        resMat[iterNum, 98] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 173] <- MSparams$rho # cardinality of support
        resMat[iterNum, 212] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 226] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, betasMS, zAvg, res)
        ########################################################
        # 
        ####################################
        # MS4 -- different support with beta - betaBar penalty AND NO frobenius norm AND NO z- zbar penalty
        ####################################
        # share info on the beta - betaBar but no ||z - zbar|| penalty (currently but could do it if changed tuning grid)
        print(paste("iteration: ", iterNum, " Bbar"))
        # *************************
        # original 1/20/22
        if(tuneInd){
          
          tune.grid_beta <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
          colnames(tune.grid_beta) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          # order correctly
          tune.grid_beta <- tune.grid_beta[  order(-tune.grid_beta$rho,
                                                   -tune.grid_beta$lambda2,
                                                   decreasing=TRUE),     ]
          
          tune.grid_beta <- unique(tune.grid_beta)
          # if tune in two stages
          print(paste("iteration: ", iterNum, " Bbar Tune 1"))
          
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
                                     maxIter = maxIter_cv,
                                     threads = tuneThreads,
                                     WSmethod = WSmethod,
                                     ASpass = ASpass
          )
          
          MSparams <- tuneMS$best # parameters
          rhoStar <- MSparams$rho
          lambdaBstar <- MSparams$lambda2
          
          rhoG <- rho # (rhoStar - 1):(rhoStar + 1)
          lambdaBgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaBstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
          lambdaBgrid <- sort(lambdaBgrid, decreasing = TRUE)
          
          gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBgrid, 0, rhoG) )
          colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")
          
          gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                           -gridUpdate$lambda2,
                                           -gridUpdate$lambda_z,
                                           decreasing=TRUE),     ]
          
          gridUpdate <- unique(gridUpdate)
          
          print(paste("iteration: ", iterNum, " Bbar Tune 2"))
          
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
                                     ASpass = ASpass
          )
          
        }else{
          # tune in 1 stage
          
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
                                     maxIter = maxIter_cv,
                                     threads = tuneThreads,
                                     WSmethod = WSmethod,
                                     ASpass = ASpass
          )
        }
        
        MSparams <- tuneMS$best # tuned parameters
        
        # warm start
        print(paste("iteration: ", iterNum, " Bbar WS"))
        
        warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                            y = as.matrix( full[, Yindx] ),
                            rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = b,
                            lambda1 = MSparams$lambda1,
                            lambda2 = 0, #MSparams$lambda2,
                            lambda_z = MSparams$lambda_z,
                            scale = TRUE,
                            maxIter = 10000,
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
                          maxIter = 10000,
                          localIter = localIters,
                          WSmethod = WSmethod,
                          ASpass = ASpass
        )
        
        
        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
        
        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
          supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }
        
        resMat[iterNum, 68:71] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 99] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 175] <- MSparams$rho # cardinality of support
        resMat[iterNum, 214] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 228] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        lambdaBbarstar <- MSparams$lambda2
        rm(MSparams, tuneMS, supMat, betasMS, zAvg, warmStart)
        ########################################################
        
        ####################################
        # MS5 -- beta - betaBar AND ||z - zbar|| 
        ####################################
        # share info on the beta - betaBar AND ||z - zbar|| 
        print(paste("iteration: ", iterNum, " zBar/bBar"))
        timeStart1 <- Sys.time()
        
        glPenalty <- 5
        
        if(tuneInd){
          
          # ************
          # original from 1/20/21
          # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
          # ************
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
                                     ASpass = ASpass
          )
          
          MSparams <- tuneMS$best # parameters
          rhoStar <- MSparams$rho
          lambdaZstar <- MSparams$lambda_z
          lambdaBstar <- MSparams$lambda2
          
          rhoG <- rho # (rhoStar - 1):(rhoStar + 1)
          lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
          lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
          
          # ************
          # original from 1/20/21
          # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
          # ************
          gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBstar, lambdaZgrid, rhoG) ) # 1e-8
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
                                     ASpass = ASpass
          )
          
        }else{
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, lambda, lambdaZ, rho) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                                       -tune.grid_MSZ_5$lambda2,
                                                       -tune.grid_MSZ_5$lambda_z,
                                                       decreasing=TRUE),     ]
            
            tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)
            
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
        ###################################################
        print(paste("iteration: ", iterNum, " zBar/bBar WS"))
        
        # warm start
        warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                          y = as.matrix( full[, Yindx] ),
                          rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = b,
                          lambda1 = MSparams$lambda1,
                          lambda2 = MSparams$lambda2,
                          lambda_z = MSparams$lambda_z,
                          scale = TRUE,
                          maxIter = 10000,
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
                          maxIter = 10000,
                          localIter = localIters ,
                          WSmethod = WSmethod,
                          ASpass = ASpass
        )
        
        timeEnd1 <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 232] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))


        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }

        resMat[iterNum, 189:192] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 199] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 200] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 201] <- MSparams$rho # cardinality of support
        resMat[iterNum, 213] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 227] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, supMat, betasMS, zAvg, warmStart)
        
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
                               threads = tuneThreads)

        MSparams <- tuneMS$best # parameters

        # warm start
        warmStart = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                        y = as.matrix( full[, Yindx] ),
                        rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = b,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
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
                        maxIter = 10000,
                        localIter = localIters
        )
        
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j] )
        }

        resMat[iterNum, 78:81] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 100] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 177] <- MSparams$rho # cardinality of support
        resMat[iterNum, 215] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 229] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, supMat, betasMS, zAvg, warmStart)
        ######################################################## 
        ####################################
        # Convex MS, glPenalty = 3, IP FALSE: Just Frobenius norm and no other sharing of information
        ####################################
        print(paste("iteration: ", iterNum, " Frobenius only"))
        
        glPenalty <- 3
        ip <- FALSE
        tune.grid2 <- tune.grid
        tune.grid2$rho <- length(Xindx) # full support
        tune.grid2 <- unique(tune.grid2)
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix

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

        betasMS = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                        y = as.matrix( full[, Yindx] ),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = betas,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 0 # convex
        )

        zAvg <- I( rowMeans(betasMS)[-1] != 0) * 1 # stacking
        resMat[iterNum, 101] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 179] <- MSparams$rho # cardinality of support
        resMat[iterNum, 216] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 230] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        
        rm(MSparams, tuneMS, zAvg)
        ########################################################

        ########################################################
        # ***multi-task learning***
        ########################################################
        print(paste("iteration: ", iterNum, " RMTL"))
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
        
        lambda_vector <- sort( unique( c(0.0001, 0.001, 0.01, 5,10, 50, 100, 200,
                                         exp(-seq(0,5, length = tune_length))) ), 
                               decreasing = TRUE ) 

        ##############
        # RMTL - Trace
        ##############
        # format data for multi-task learning
        
        cvfitc <- cvMTL(X = Xlist, 
                        Y = Ylist,
                        type="Regression", 
                        Regularization="Trace", 
                        Lam1_seq = lambda_vector,
                        Lam2 = 0,
                        nfolds = nfold, 
                        parallel = FALSE)
        
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
        
        resMat[iterNum, 109] <-  sum( model$W[-1,] != 0 ) / K # size of xupport
        resMat[iterNum, 110] <- multiTaskRmse_MT(data = mtTest, beta = model$W)
        
        rm(model, cvfitc)
        ########################################################
        ########################################################
        #######################
        # sparse group lasso
        #######################
        print(paste("iteration: ", iterNum, " sGL"))
        
        # convert into form used by `sparsegl` package
        Xlist <- lapply(Xlist, function(x) as.matrix(x[,-1]) ) # make matrix and remove column of 1s for intercept
        Xlist <- Matrix::bdiag(Xlist)
        Ymeans <- colMeans(full[,Yindx]) # means for intercept
        Ylist <- lapply(Ylist, function(x) scale(x, center = TRUE, scale = FALSE)) # center Ys for each task
        Ylist <- as.numeric(do.call(c, Ylist))
        groupIDs <- rep(1:numCovs, K)
        
        # adjust foldID to align with glmnet above
        foldID <- rep(foldID, K)
        
        # hyperparameter tuning
        num_alpha <- 10
        asprse_mat <- data.frame(matrix(NA, nrow = num_alpha, ncol = 3))
        colnames(asprse_mat) <- c("rmse", "lambda", "alpha")
        asprse_mat$alpha <- seq(0,1, length = num_alpha) # 2nd hyperparameter
        beta_matrix <- matrix(NA, ncol = num_alpha, nrow = ncol(Xlist) + 1) # store best models
        
        for(asprs in 1:nrow(asprse_mat)){
          print(asprs)
          # tune model
          tuneMS <- cv.sparsegl(x = Xlist,
                                y = Ylist,
                                foldid = foldID,
                                group = groupIDs,
                                intercept = FALSE,
                                nlambda = tune_length*50, # ensure enough values at the specific rho value
                                asparse = asprse_mat$alpha[asprs])
          
          sprs_const <- sapply(tuneMS$lambda, function(x) sum(coef(tuneMS, s = x) != 0 ) / K) # which coefs to be considered by sparsity
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
            
          min_idx <- which.min(tuneMS$cvm[sprs_const])
          asprse_mat$rmse[asprs] <- min(tuneMS$cvm[sprs_const])
          asprse_mat$lambda[asprs] <- (tuneMS$lambda[sprs_const])[min_idx] # save results
          beta_matrix[,asprs] <- as.numeric(coef(tuneMS, s = asprse_mat$lambda[asprs] ))
          
        }
        
        # tuned hyperparameters
        asparse <- asprse_mat$alpha[which.min(asprse_mat$rmse)]
        lambdaMin <- asprse_mat$lambda[which.min(asprse_mat$rmse)]
        min_index <- which.min(asprse_mat$rmse)
        
        beta_sprgl <- beta_matrix[-1, min_index ]
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        # results
        curIdx <- 235
        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
        
        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
          supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }
        
        resMat[iterNum, curIdx+1] <- lambdaMin # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+3] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, curIdx+4] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        resMat[iterNum, (curIdx + 5):(curIdx + 8)] <- colMeans(supMat)
        resMat[iterNum, curIdx + 9] <- asparse # alpha
        curIdx <- curIdx + 9 # update
        rm(beta_sprgl, lambdaMin, idxMatrix)
        ########################################################
        ########################
        # group lasso
        ########################
        print(paste("iteration: ", iterNum, " GL not-sparse"))
        
        # hyperparameter tuning
        tuneMS <- cv.sparsegl(x = Xlist,
                              y = Ylist,
                              foldid = foldID,
                              group = groupIDs,
                              intercept = FALSE,
                              nlambda = tune_length*50, # ensure enough values at the specific rho value
                              asparse = 0)
        
        sprs_const <- sapply(tuneMS$lambda, function(x) sum(coef(tuneMS, s = x) != 0 ) / K) # which coefs to be considered by sparsity
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
        
        min_idx <- which.min(tuneMS$cvm[sprs_const])
        lambdaMin <- (tuneMS$lambda[sprs_const])[min_idx] # save results 
        
        beta_sprgl <- as.numeric(coef(tuneMS, s = lambdaMin)) 
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
        
        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
          supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }
        
        resMat[iterNum, curIdx+1] <- lambdaMin# MSparams$lambda # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+3] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, curIdx+4] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        resMat[iterNum, (curIdx + 5):(curIdx + 8)] <- colMeans(supMat)
        resMat[iterNum, curIdx + 9] <- 0
        curIdx <- curIdx + 9 # update
        curIdx_orig <- curIdx # use this so we can switch back to counting after standard lasso below
        
        rm(beta_sprgl, lambdaMin, idxMatrix)
        ########################################################
        ########################
        # lasso
        ########################
        curIdx <- 280
        print(paste("iteration: ", iterNum, " lasso"))
        
        # hyperparameter tuning
        tuneMS <- cv.sparsegl(x = Xlist,
                              y = Ylist,
                              foldid = foldID,
                              group = groupIDs,
                              intercept = FALSE,
                              nlambda = tune_length*50, # ensure enough values at the specific rho value
                              asparse = 1)
        
        sprs_const <- sapply(tuneMS$lambda, function(x) sum(coef(tuneMS, s = x) != 0 ) / K) # which coefs to be considered by sparsity
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
        
        min_idx <- which.min(tuneMS$cvm[sprs_const])
        lambdaMin <- (tuneMS$lambda[sprs_const])[min_idx] # save results 
        
        beta_sprgl <- as.numeric(coef(tuneMS, s = lambdaMin)) 
        
        # make into matrix
        idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
        beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
        betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
        
        # support stats
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
        
        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
          supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
        }
        
        resMat[iterNum, curIdx+1] <- lambdaMin# MSparams$lambda # tuning parameter
        resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
        resMat[iterNum, curIdx+3] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, curIdx+4] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
        resMat[iterNum, (curIdx + 5):(curIdx + 8)] <- colMeans(supMat)
        resMat[iterNum, curIdx + 9] <- 1
        curIdx <- curIdx_orig # switch back to counting indices
        
        rm(beta_sprgl, lambdaMin, idxMatrix)
        ########################################################
        ########################
        # grpreg
        ########################
        print(paste("iteration: ", iterNum, " grpreg"))
        
        grp_penalties <- c("grMCP",  "gel", "cMCP")
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
                              fold = foldID,
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
          #}
          rm(tuneMS)
          
          # tuned hyperparameters
          lambdaMin <- asprse_mat$lambda
          beta_sprgl <- beta_vec[-1] # remove intercept since it will be approx 0
          
          # make into matrix
          idxMatrix <- t(sapply( seq(1, K*numCovs, by = numCovs), function(x) seq(x,x+numCovs-1)))
          beta_sprgl <- apply(idxMatrix, 1, function(x) beta_sprgl[x]) 
          betasMS <- rbind(Ymeans, beta_sprgl) # add intercepts back on
          
          # support stats
          zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
          
          supMat <- matrix(NA, nrow = K, ncol = 4)
          for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
          }
          
          resMat[iterNum, curIdx+1] <- lambdaMin # tuning parameter
          resMat[iterNum, curIdx+2] <- mean(colSums(betasMS[-1,] != 0)) # average cardinality of support
          resMat[iterNum, curIdx+3] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
          resMat[iterNum, curIdx+4] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
          resMat[iterNum, (curIdx + 5):(curIdx + 8)] <- colMeans(supMat)
          resMat[iterNum, curIdx + 9] <- 1 # alpha
          curIdx <- curIdx + 9 # update
          
          rm(beta_sprgl, lambdaMin, idxMatrix)
          
        }
        ########################################################
        
        
        ########################################################
        
        print(paste("iteration: ", iterNum, " Complete!"))
        # print(resMat[iterNum,])
        
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
