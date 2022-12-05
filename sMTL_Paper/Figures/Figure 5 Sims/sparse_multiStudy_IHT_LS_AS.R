# Updates: December 19, 2021 
# uses active set versions appropriate for each method
library(pROC)
library(JuliaConnectoR)
library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)

source("SimFn.R")
source("OEC Functions.R")

# # # s is true number of non-zeros
# # r is number of covariates that are potentially divergent support (set of covariates that are randomly selected to be include)
# # r_p is the probability of inclusion for a given study (i..e, each of the K studies selects each of the r covariates for inclusion ~ i.i.d. Ber(r_p) )

sims6 <- read.csv("sparseParam_test")
cluserInd <- TRUE # whether running on computer or on cluster

save.folder <- "/n/home12/gloewinger/sparse22"
load.folder <- "~/Desktop/Research"
if(cluserInd){
    # if on cluster
    args = commandArgs(TRUE)
    runNum <- as.integer( as.numeric(args[1]) )
    iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id

    # Julia paths
    juliaPath <- "/n/sw/eb/apps/centos7/Julia/1.5.3-linux-x86_64/bin"
    juliaFnPath <- "/n/home12/gloewinger/"
}else{
    runNum <- 1
    setwd("~/Desktop/Research")
    iterNum <- 1

    # Julia paths
    juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
    juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune/"
    juliaFnPath_MT <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune MT/"
    

}

Sys.setenv(JULIA_BINDIR = juliaPath)

# sim params
simNum <- runNum
totalSims <- 30

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
 
p <- 0
q <- 5
numCovs <- sims6$p[runNum] #p + q + 45 # p and q here refer to non-zero coefficients
s <- sims6$s[runNum] #
r <- sims6$r[runNum]
r_p <- sims6$r_p[runNum]
zeroCovs <- seq(2, numCovs + 1)[-seq(2, 2 * s, by = 2)] # alternate because of exponential correlation structure
# Study Strap and Model fitting parameters
test_study <- max(trainStudies) + 1 # arbitrarily set to study but any of the non-training studies is fine (11-24 are all random test studies)
scaleInd <- TRUE
betaMeanRange <- c(sims6$betaRangeLow[runNum],   sims6$betaRangeHigh[runNum])
clustDiv <- 10
Mmultiplier <- 1.5
timeLimTn <- 60 # time limit for tuning
timeLim <- 1200 # time limit for running
seedFixedInd <- TRUE # fixed effects (true betas) and Sigma_x fixed across simulation iterations
covType <- sims6$cov[runNum] # type of covariance matrix for the features
rho_corr <- sims6$rho[runNum] # rho used in covariance matrix for features if "exponential" or "pairwaise"
MoM <- FALSE # indicator of whether to run MoM
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- 50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- 1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 50 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning

errorMult <- sims6$errorMult[runNum] # range of error for uniform
tau <- sims6$tau[runNum]
epsHigh <- tau * errorMult# noise lower/upper
epsLow <- tau / errorMult# noise lower/upper
nLow <- nHigh <- sims6$nLow[runNum]  # multiply by 2 because of multi-task test set  # samp size lower/upper
tuneInd <- sims6$tuneInd[runNum]
WSmethod = 2 # sims6$WSmethod[runNum]
ASpass = TRUE # sims6$ASpass[runNum]

if(tuneThreads == 1){
    # use non-parallel version
    source("sparseFn_iht_test.R") # USE TEST VERSION HERE
    sparseCV_iht_par <- sparseCV_iht
}else{
    # source("sparseFn_iht_par.R")
    source("sparseFn_iht_test.R") # USE TEST VERSION HERE
    sparseCV_iht_par <- sparseCV_iht
}

# model tuning parameters
L0TuneInd <- TRUE # whether to retune lambda and rho with gurobi OSE (if FALSE then use L0Learn parameters)
L0MrgTuneInd <- TRUE # whether to retune lambda and rho with gurobi Mrg (if FALSE then use L0Learn parameters)
L0_sseTn <- "sse" # tuning for L0 OSE (with gurobi not L0Learn)
MSTn <- sims6$multiTask[runNum] #"hoso" #"balancedCV" # tuning for MS (could be "hoso")

if(MSTn %in% c("hoso", "balancedCV") ){
    # if not multi-task then this can't be smaller than K
    nfold <- min( 5, K) # 5 fold maximum
}else if(MSTn == "multiTask"){
    # if multi-task then can do 5 fold CV since we are not doing a hold-one-study-out CV
    nfold <- 5
}

nfoldL0_ose <- min( 5, K) # 5 fold maximum


lamb <- 0.5
fileNm <- paste0("sprsMS_LS",
                "_s_", s, "_r_", r, "_rp_", r_p,
                  "_numCovs_", numCovs, "_n_", nLow, ".", nHigh,
                 "_eps_", epsLow, ".", epsHigh,
                "_covTyp_", covType,
                "_rho_", rho_corr,
                "_clustDiv_", clustDiv,
                "_bVar_", betaVar, "_xVar_",
                round( covVar, 2), "_clst_",
                length(unique(clusts)) - 1, "_K_", K,
                "_bMean_", betaMeanRange[1], "_", betaMeanRange[2], "_",
                "_bFix_", seedFixedInd,
                "_L0sseTn_", L0_sseTn,
                "_MSTn_", MSTn,
                "_nFld_", nfold,
                "_LSitr_", LSitr,
                "_LSspc_", LSspc,
               "_wsMeth_", WSmethod,
               "_asPass_", ASpass,
               "_TnIn_", tuneInd)
print(fileNm)

minRho <- max(   c(1, (s - 2) )   )
maxRho <- min(   numCovs,  (s+r+1)  )
rho <- minRho:maxRho

lambda <- sort( unique( c(0, 0.0001, 0.001, 0.01, 5,10, 50, 100, 200,
                          exp(-seq(0,5, length = 15))
                          #seq(120, 200, by = 20)
) ), decreasing = TRUE ) # 2:100

lambdaShort <- sort( unique( c(0,
                               exp(-seq(0,5, length = 5)),
                               5,10, 50, 100, 250, 500, 1000, 2500, 5000, 10000) ),

                     decreasing = TRUE ) # 2:100
lambdaZ <- sort( unique( c(0, 1e-6, 1e-5, 1e-4, 1e-3,
                               exp(-seq(0,5, length = 8)),
                               1:3) ),
                     decreasing = TRUE ) # 2:100

lambdaBeta <- c( 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 )

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
                                            c(lambda) ,
                                            rho)
                             ) # tuning parameters to consider
colnames(tune.grid) <- c("lambda", "rho")

# glmnet for ridge
tune.gridGLM <- as.data.frame( cbind(0, lambda) ) # Ridge
colnames(tune.gridGLM) <- c("alpha", "lambda")

timeStart <- Sys.time()

#####################################################################
print(paste0("start: ", iterNum))


    # read in sparse functions from Julia

    Sys.setenv(JULIA_BINDIR = juliaPath)
    L0_reg <- juliaCall("include", paste0(juliaFnPath, "l0_IHT_tune.jl") ) # sparseReg # MT: doesnt make sense
    L0_MS <- juliaCall("include", paste0(juliaFnPath, "BlockIHT_tune.jl") ) # MT: Need to check it works
    L0_MS2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_tune.jl") ) # MT: Need to check it works;   multi study with beta-bar penalty
    L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
    L0_MS_z2 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_tuneTest.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
    L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study
    
    # MoM
    MoM_L0 <- juliaCall("include", paste0(juliaFnPath, "MoM_IHT_tune.jl") ) # max eigenvalue

    # gurobiEnviornment

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
                          "MoM_mrg", "MoM_mrg_fp", "MoM_mrg_tp",
                          "MoM_mrg_sup", "MoM_mrg_auc", "MoM_mrg_rho",
                          ##
                          "MoM_mrg_L0", "MoM_mrg_L0_fp", "MoM_mrg_L0_tp",
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
                                 
                          ), "time1", "time2", "time3", "totalTime", "timeRidge"
                          )


        # save results
        #setwd(save.folder)
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
            suppRandom <- rbinom(r, 1, prob = r_p) # iid bernoulli draws that is r long
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
        rfB <- fixB

        # simulate data
        full <- multiStudySimNew(
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
                               corr_rho = rho_corr # used if pariwise or exponential correlation
                               )

        ### SNR
        #-----------------------------
        trueZ <- I(full$betas != 0) * 1
        trueB <- t( full$betas )[,-test_study] # true betas
        #-----------------------------

        full <- as.data.frame( full$data )
        
        # split multi-study
        mtTest <- multiTaskSplit(data = full, split = 0.5) # make training sets
        full <- mtTest$train
        mtTest <- mtTest$test
        # vector of true indicators of whether 0 or not
        
        z <- rep(1, numCovs)
        z[zeroCovs - 1] <- 0

        ## test
        test_country <- setdiff(1:studyNum, trainStudies ) # indx of test country
        testIndx <- which(full$Study == test_country)
        
        # train test for domain generalization
        test <- full[testIndx,]
        full <- full[-testIndx,]  # remove rows corresponding to testing
        rm(testIndx)
        
        # remove study (K+1) from multi-task TEST dataset
        testIndx <- which(mtTest$Study == test_country)
        mtTest <- mtTest[-testIndx,] # remove observations for mtTest associated with "test" study (K+1) since these are not used for multi-task learning
        rm(testIndx)
        
        countries <- unique(full$Study) # only include countries that have both
        K <- length(countries) # number of training countries

        #################
        # Study Labels
        #################
        countries <- unique(full$Study) # number corresponding to it
        #
        # X <- full[,-c(1,2)]
        # X_test <-  test[,-1]

        ####################################
        # scale covariates
        ####################################

        if(scaleInd == TRUE){

            nFull <- nrow( full ) # sample size of merged

            # scale Covaraites
            means <- colMeans( as.matrix( full[,-c(1,2)]  ) )
            sds <- sqrt( apply( as.matrix( full[,-c(1,2)]  ), 2, var) *  (nFull - 1) / nFull )  # use mle formula to match with GLMNET

            # columns 1 and 2 are Study and Y
            for(column in 3:ncol(full) ){
                # center scale
                full[, column] <- (full[, column ] - means[column - 2]) / sds[column - 2]
                test[, column] <- (test[, column ] - means[column - 2]) / sds[column - 2]
                mtTest[, column] <- (mtTest[, column ] - means[column - 2]) / sds[column - 2]
                
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
                X_test[, column] <- (X_test[, column ] - means[column]) / sds[column]
                test2[, column + 2] <- (test2[, column + 2] - means[column]) / sds[column]
            }

            # update the design matrix
            full[,-c(1,2)] <- X
            test[,-1] <- X_test

        }

        test <- test[,-1] # remove Study labels from test set
        mtTest <- mtTest  # remove study labels from multi-task test set
        
        rownames(mtTest) <- 1:nrow(mtTest)
        rownames(full) <- 1:nrow(full)
        rownames(test) <- 1:nrow(test)
        
        timeStartTotal <- Sys.time()
        #######################
        # Tuning for Ridge
        #######################
        # # Merge Tune for Ridge
        print(paste("iteration: ", iterNum, " Tune Ridge Meg/OSE"))
        mergedLambda <- hosoCV(data = full,
                               tune.gridGLM,
                               hoso = "merged",
                               method = "glmnet",
                               metric = "RMSE",
                               nfolds = nfold )

        #######################
        # Tuning for LASSO
        #######################
        # # Merge Tune for LASSO
        tune.gridGLM2 <- data.frame(alpha = 1, lambda = tune.gridGLM$lambda) # set alpha = 1 for lasso
        print(paste("iteration: ", iterNum, " Tune Ridge Meg/OSE"))
        mergedLambdaLasso <- hosoCV(data = full,
                               tune.gridGLM2,
                               hoso = "merged",
                               method = "glmnet",
                               metric = "RMSE",
                               nfolds = nfold )


        ##############
        # Merge Ridge
        ##############
        # print(paste0("Merge: ", iterNum))

        mrg.mod <- glmnet(y = full$Y,
                          x = as.matrix(full[,-c(1,2)]),
                          alpha = mergedLambda$alpha,
                          lambda = mergedLambda$lambda,
                          intercept = TRUE)

        betaEst <- coef(mrg.mod, exact = TRUE)
        betaEst <- betaEst %*% t(rep(1, K))
        preds <- predict(mrg.mod, as.matrix(test[,-1]) ) # predict
        resMat[iterNum, 1] <- sqrt( mean( (test$Y - preds)^2 ) ) # rmse
        resMat[iterNum, 203] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 217] <- multiTaskRmse(data = mtTest, beta = betaEst)
        rm(mrg.mod, betaEst)

        ##############
        # Merge Lasso
        ##############
        # print(paste0("Merge: ", iterNum))

        mrgLasso <- glmnet(y = full$Y,
                          x = as.matrix(full[,-c(1,2)]),
                          alpha = mergedLambdaLasso$alpha,
                          lambda = mergedLambdaLasso$lambda,
                          intercept = TRUE)

        betaEst <- coef(mrgLasso, exact = TRUE)
        betaEst <- betaEst %*% t(rep(1, K))
        
        preds <- predict(mrgLasso, as.matrix(test[,-1]) ) # predict
        resMat[iterNum, 102] <- sqrt( mean( (test$Y - preds)^2 ) ) # rmse

        lassoBeta <- as.vector( coef(mrgLasso) )
        lassoSupp <- I( lassoBeta != 0) * 1

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = lassoSupp[-1])
        }

        resMat[iterNum, 103:106] <- colMeans(supMat) # 
        resMat[iterNum, 141] <- suppStat(response = trueZ[K + 1, -1], predictor = lassoSupp[-1])[3] # support recovery of test study
        resMat[iterNum, 162] <-  sum(lassoSupp[-1]) # size of xupport
        resMat[iterNum, 204] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 218] <- multiTaskRmse(data = mtTest, beta = betaEst)
        
        rm(mrgLasso, lassoSupp, lassoBeta, supMat, betaEst)

        ##############
        # OSE Ridge
        ##############
        timeStart2 <- Sys.time()
        print(paste("iteration: ", iterNum, " Ridge OSE"))
        b <- matrix(0, ncol = K, nrow =numCovs + 1)
        
        tune.grid_sse <- data.frame(lambda1 = unique(lambda),
                                    lambda2 = 0,
                                    lambda_z = 0,
                                    #rho = numCovs)
                                    rho = ncol(full) - 2)
        
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
                                    threads = tuneThreads
        )
        
        L0_tune <- L0_tune$best
        
        
        # final model
        betasMS = L0_MS_z2(X = as.matrix( full[ , -c(1,2) ]) ,
                           y = as.vector(full$Y),
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
        
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS
        
        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)
        
        w <- coef(fitW)
        rm(fitW)
        
        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        #**************FIX INDICES*************
        resMat[iterNum, 2] <- sqrt(mean( (avgPreds - test$Y)^2  ))
        resMat[iterNum, 3] <- sqrt(mean( (stackPreds - test$Y)^2  ))
        resMat[iterNum, 205] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 219] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        timeEnd2 <- Sys.time()
        resMat[iterNum, 235] <- as.numeric(difftime(timeEnd2, timeStart2, units='mins'))
        
        rm(betasMS, predsMat, testMat, avgPreds, stackPreds, w)
        ###############################################################

        ########################
        # L0Learn Mrg version
        ########################
        n <- length(full$Y)
        sdY <- 1 # set to 1 so we do NOT adjust for sd(y) since iht version does not do this # sd(full$Y) * (n - 1) / n #MLE
        gm <- tune.grid$lambda / (sdY * 2) # convert into comparable numbers for L0Learn

        cvfit = L0Learn.cvfit(x = as.matrix(full[, -c(1,2)]),
                              y = full$Y,
                              nFolds = max( c(5, K) ), # we do K-fold CV
                              seed = 1,
                              penalty="L0L2",
                              nGamma = length(gm),
                              #gammaMin = min(gm), # min and max numbers of our 2 parameter that is comaprable
                              #gammaMax = max(gm),
                              algorithm = "CDPSI",
                              maxSuppSize = max(tune.grid$rho), # largest that we search
                              scaleDownFactor = 0.99
        )

        # optimal tuning parameters
        optimalGammaIndex <- which.min( lapply(cvfit$cvMeans, min) ) # index of the optimal gamma identified previously
        optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
        optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
        L0LearnMrg <- coef(cvfit, lambda=optimalLambda, gamma = cvfit$fit$gamma[optimalGammaIndex] )

        # save parameters in case we do not retune below
        mergedL0_tune <- list() # make empty list to save optimal parameters from above
        rhoStar <- sum(  as.vector(L0LearnMrg)[-1] != 0   ) # cardinality
        mergedL0_tune$best$lambda <- cvfit$fit$gamma[optimalGammaIndex] * (2 * sdY) # put on scale used by gurobi version below
        mergedL0_tune$best$rho <- rhoStar
        betaEst <- L0LearnMrg %*% t(rep(1, K))
        
        # predictions and test error
        preds <- as.matrix( cbind(1, test[,-1] ) ) %*% L0LearnMrg # predict(fit, X_test)
        L0LearnMrg <- as.vector(L0LearnMrg)
        mrgBeta <- L0LearnMrg
        mrgZ <- I(L0LearnMrg != 0) * 1
        resMat[iterNum, 9] <- sqrt( mean( (test$Y - preds)^2 ) ) # RMSE

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = mrgZ[-1])
        }

        resMat[iterNum, 10:13] <- colMeans(supMat) # suppStat(response = z, predictor = lassoSupp[-1])
        resMat[iterNum, 142] <- suppStat(response = trueZ[K + 1, -1], predictor = mrgZ[-1])[3] # support recovery of test study
        resMat[iterNum, 163] <- mergedL0_tune$best$rho
        resMat[iterNum, 206] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 220] <- multiTaskRmse(data = mtTest, beta = betaEst)
        
        rm(sdY, preds, mrgZ, cvfit, supMat, mrgBeta, L0LearnMrg, betaEst)
        ##############################################

        #############
        # merging L0
        #############
        # tune L0 Merged
        print(paste("iteration: ", iterNum, " Mrg L0Lrn"))
        if(L0MrgTuneInd){
            mergedL0_tune <- sparseCV_iht_par(data = full,
                                      tune.grid = tune.grid,
                                      hoso = "hoso", # could balancedCV (study balanced CV necessary if K =2)
                                      method = "L0", # could be L0 for sparse regression or MS # for multi study
                                      nfolds = nfoldL0_ose, # nfoldL0_ose
                                      cvFolds = 5,
                                      juliaPath = juliaPath,
                                      juliaFnPath = juliaFnPath,
                                      messageInd = FALSE,
                                      LSitr = LSitr,
                                      LSspc = LSspc,
                                      threads = tuneThreads
            )
        }


        # intiialize randomly
        initMrg <- rep(0, ncol(full) - 1) #rnorm(ncol(full) - 1)

        fit <-  L0_reg(X = as.matrix(full[, -c(1,2)]),
                       y = full$Y,
                       rho = mergedL0_tune$best$rho,
                       beta = initMrg,
                       lambda = mergedL0_tune$best$lambda,
                       scale = TRUE,
                       maxIter = 10000,
                       localIter = 50
        )

        preds <- as.matrix( cbind(1, test[,-1] ) ) %*% (fit) # predict(fit, X_test)
        mrgBeta <- fit
        betaEst <- mrgBeta %*% t(rep(1, K))
        mrgZ <- I(fit != 0) * 1
        resMat[iterNum, 4] <- sqrt( mean( (test$Y - preds)^2 ) ) # RMSE

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = mrgZ[-1])
        }

        resMat[iterNum, 5:8] <- colMeans(supMat) # suppStat(response = z, predictor = lassoSupp[-1])
        resMat[iterNum, 143] <- suppStat(response = trueZ[K + 1, -1], predictor = mrgZ[-1])[3] # support recovery of test study
        resMat[iterNum, 164] <- mergedL0_tune$best$rho # cardinality of support
        resMat[iterNum, 207] <- sqrt( mean( (betaEst - trueB)^2 ) ) # coef error
        resMat[iterNum, 221] <- multiTaskRmse(data = mtTest, beta = betaEst)
        
        rm(fit, preds, mergedL0_tune, mrgZ, supMat, mrgBeta, betaEst)

        ##############
        # OSE L0Learn
        ##############
        print(paste("iteration: ", iterNum, " L0Learn OSE"))
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        betaMat <- betas <- matrix(NA, ncol = K, nrow = ncol(full) - 1) # matrix of betas from each study
        res <- vector(length = K) # store auc
        Mvec <- vector(length = K) # use to initialize other models later on
        # save best parameter values
        L0_tune <- matrix(NA, nrow = K, ncol = ncol(tune.grid) ) # save best parameter values
        L0_tune <- as.data.frame(L0_tune)
        colnames(L0_tune) <- colnames(tune.grid)

        supMat <- matrix(NA, nrow = K, ncol = 4)

        for(j in 1:K){

            indx <- which(full$Study == j) # rows of each study
            n_k <- length(indx) # sample size of jth study
            sdY <- 1 # set to 1 for now so we DO NOT adjust as glmnet() #sd(full$Y[indx]) * (n_k - 1) / n_k #MLE
            gm <- tune.grid$lambda / (sdY * 2) # convert into comparable numbers for L0Learn

            # fit l0 model on jth study
            cvfit = L0Learn.cvfit(x = as.matrix(full[indx, -c(1,2)]),
                                  y = as.vector(full$Y[indx]),
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
            predsMat[,j] <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% L0LearnCoef

            # test set prediction matrix
            testMat[,j] <- as.matrix( cbind(1, test[,-1] ) ) %*% L0LearnCoef # predict on test set

            # model specific nonzero covs
            zOSE <- I(L0LearnCoef != 0) * 1 # nonzero betas

            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zOSE[-1])
            rm(cvfit, indx)
        }


        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                      x = as.matrix(predsMat),
                      alpha = 0,
                      lambda = 0,
                      standardize = TRUE,
                      intercept = TRUE,
                      thresh = 1e-10,
                      lower.limits = 0)

        w <- coef(fitW)
        rm(fitW, sdY)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 14] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 15] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # support stats
        zStack <- I( betaMat %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( rowMeans(betaMat)[-1] != 0) * 1 # stacking

        res <- suppStat(response = trueZ[K + 1, -1], predictor = zAvg )[3]

        resMat[iterNum, 16:19] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 144] <- res # avg support prediction

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }

        resMat[iterNum, 20:23] <- colMeans(supMat) # suppStat(response = z, predictor = lassoSupp[-1])
        resMat[iterNum, 145] <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3] # support recovery of test study
        resMat[iterNum, 165] <- mean(L0_tune$rho) # cardinality of support
        resMat[iterNum, 166] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 208] <- sqrt( mean( (betaMat - trueB)^2 ) ) # coef error
        resMat[iterNum, 222] <- multiTaskRmse(data = mtTest, beta = betaMat)
        
        rm(betaMat, w, res, supMat, zStack, zAvg, zOSE)

        ##############################################


        ############
        # OSE L0
        ############
        timeStart1 <- Sys.time()
        
        tune.grid_OSE <- data.frame(lambda1 = unique(lambda),
                                    lambda2 = 0,
                                    lambda_z = 0,
                                    #rho = numCovs)
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
                                    WSmethod = WSmethod,
                                    ASpass = ASpass
                                    )
        
        L0_tune <- L0_tune$best # parameters
        
        # initialize algorithm warm start
        p <- ncol(full) - 1
        b <-  matrix(0, ncol = K, nrow = p ) #rep(0, p) #rnorm( p )
        
        # matrices
        betas <- matrix(NA, nrow = p, ncol = K)
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
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
                                     localIter = 50,
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
        
        resMat[iterNum, 233] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        
        for(j in 1:K){
            
            indx <- which(full$Study == j) # rows of each study
            
            fitG <- betas[, j]
            
            ##########################################
            # use Gurobi estimates as warm starts (not L0Learn)
            ##########################################

            # stack matrix
            predsMat[,j] <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% fitG
            
            # test set prediction matrix
            testMat[,j] <- as.matrix( cbind(1, test[,-1] ) ) %*% fitG # predict on test set
            
            # model specific nonzero covs
            zOSE <- I(fitG != 0) * 1 # nonzero betas
            
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zOSE[-1])
            rm(indx)
            
        }

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions %*% w # stacking predictions ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 25] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 26] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # stacking auc
        zStack <- I( betas %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( rowMeans(betas)[-1] != 0) * 1# stacking

        res <- suppStat(response = trueZ[K + 1, -1], predictor = zAvg )[3]

        resMat[iterNum, 27:30] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 146] <- res # average auc of each study-specific model

        supMat <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }

        resMat[iterNum, 31:34] <- colMeans(supMat) #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 147] <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3] # support recovery of test study
        resMat[iterNum, 167] <- L0_tune$rho # cardinality of support
        resMat[iterNum, 168] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 209] <- sqrt( mean( (betas - trueB)^2 ) ) # coef error
        resMat[iterNum, 223] <- multiTaskRmse(data = mtTest, beta = betas)
        
        rm(w, res, L0_tune, supMat, zOSE, zAvg, zStack, betas)

        ####################################
        # common support L0 regularization with ||beta - betaBar|| penalty
        # glPenalty = TRUE, ip = TRUE
        ####################################
        # random initialization for betas
        betas <- matrix( 0, nrow = numCovs + 1, ncol = K )#matrix( rnorm( K * p ), ncol = K )

        print(paste("iteration: ", iterNum, " Group IP"))
        glPenalty <- 1
        ip <- TRUE
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
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
                           messageInd = FALSE,
                           LSitr = LSitr,
                           LSspc = LSspc,
                           threads = tuneThreads
        )

        MSparams <- tuneMS$best # parameters

        # warm start for 4 * rho and no betaBar regularization
        warmStart = L0_MS2(X = as.matrix( full[ , -c(1,2) ] ),
                         y = as.vector(full$Y),
                         rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = betas,
                         lambda1 = MSparams$lambda1,
                         lambda2 = 0, # use 0 as warm start
                         scale = TRUE,
                         maxIter = 10000,
                         localIter = 50
        )

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
                      localIter = 50
        )

        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 36] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 37] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # support stats
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
            supMatS[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }

        res <- suppStat(response = trueZ[K + 1, -1], predictor = rowMeans(zAvg) )[3]
        resS <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3]

        resMat[iterNum, 38:41] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 42:45] <- colMeans(supMatS) # suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 148] <- res
        resMat[iterNum, 149] <- resS
        resMat[iterNum, 96] <- MSparams$lambda2 # tuning parameter
        
        resMat[iterNum, 169] <- MSparams$rho # cardinality of support
        resMat[iterNum, 170] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 210] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 224] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, supMat, supMatS, res, resS, zAvg, zStack, betasMS, warmStart)

        ########################################################

        ####################################
        #  L0 regularization with ||z - zbar|| penalty and (potentially) ||beta - betaBar|| and (potentially) frobenius norm
        # glPenalty = 2
        ####################################
        timeStart1 <- Sys.time()
        
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        res <- resS <- vector(length = K) # store support prediction
        
        if(tuneInd){
            
            # ************
            # original from 1/20/21
            # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
            # ************
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
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
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            rhoG <- (rhoStar - 1):(rhoStar + 1)
            lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # original from 1/20/21
            # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
            # ************
            gridUpdate <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZgrid, rhoG) )
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
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
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
                                       ASpass = ASpass
            )
        }
        

        MSparams <- tuneMS$best # parameters

        # warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
        warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                          y = as.vector(full$Y),
                          rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = betas,
                          lambda1 = MSparams$lambda1,
                          lambda2 = MSparams$lambda2,
                          lambda_z = 0,
                          scale = TRUE,
                          maxIter = 10000,
                          localIter = 50,
                          WSmethod = WSmethod,
                          ASpass = ASpass
        )

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
                         ASpass = ASpass
        )
        
        timeEnd1 <- timeEnd <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 231] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
        

        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 46] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 47] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # support stats
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
            supMatS[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }

        res <- suppStat(response = trueZ[K + 1, -1], predictor = rowMeans(zAvg))[3]
        resS <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3]

        resMat[iterNum, 48:51] <- colMeans(supMat) #suppStat(response = z,  predictor = zAvg * 1 )  # auc average weights ose
        resMat[iterNum, 52:55] <- colMeans(supMatS) #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 150] <- res
        resMat[iterNum, 151] <- resS
        resMat[iterNum, 97] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 171] <- MSparams$rho # cardinality of support
        resMat[iterNum, 172] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 211] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 225] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, supMat, supMatS, res, resS, betasMS, zAvg, zStack, warmStart )
        ########################################################
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
        tune.grid_MS2$rho <- ncol(full) - 2 # full support
        tune.grid_MS2 <- unique(tune.grid_MS2) # convex -- no IP selection

        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
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
                               messageInd = FALSE,
                               LSitr = NA,
                               LSspc = NA, # convex
                               threads = tuneThreads 
        )

        MSparams <- tuneMS$best # parameters

        betasMS = L0_MS2(X = as.matrix( full[ , -c(1,2) ]) ,
                         y = as.vector(full$Y),
                         rho = MSparams$rho,
                         study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                         beta = betas,
                         lambda1 = MSparams$lambda1,
                         lambda2 = MSparams$lambda2,
                         scale = TRUE,
                         maxIter = 10000,
                         localIter = 0 # convex
        )


        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 56] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 57] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # stacking auc
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( rowMeans(betasMS)[-1] != 0) * 1 # stacking

        resMat[iterNum, 58:61] <- suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 62:65] <- suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 98] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 173] <- MSparams$rho # cardinality of support
        resMat[iterNum, 174] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 212] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 226] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, betasMS, zAvg, zStack, res)
        ########################################################
        #
        ####################################
        # MS5 -- different support with beta - betaBar penalty AND z- zbar penalty, no frobenius
        ####################################
        # share info on the beta - betaBar AND ||z - zbar|| 
        timeStart1 <- Sys.time()
        
        glPenalty <- 4
        
        # ************************************************
        
        if(tuneInd){
            
            # ************
            # original from 1/20/21
            # tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) ) # tuning parameters to consider
            # ************
            
            tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, 0, lambdaZ, rho) )
            colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
            
            # order correctly
            tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
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
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            rhoG <- (rhoStar - 1):(rhoStar + 1)
            lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # ************
            gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBeta, lambdaZgrid, rhoG) )
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
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
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
                                       ASpass = ASpass
            )
        }
        
        MSparams <- tuneMS$best # parameters
        ###################################################
        # warm start
        warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                          y = as.vector(full$Y),
                          rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                          study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                          beta = betas,
                          lambda1 = MSparams$lambda1,
                          lambda2 = 0, #MSparams$lambda2,
                          lambda_z = MSparams$lambda_z,
                          scale = TRUE,
                          maxIter = 10000,
                          localIter = 50,
                          WSmethod = WSmethod,
                          ASpass = ASpass
        )

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
                          ASpass = ASpass
        )
        
        timeEnd1 <- Sys.time()
        
        print(difftime(timeEnd, timeStart, units='mins'))
        resMat[iterNum, 232] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))

        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 187] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 188] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # support stats
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
            supMatS[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }

        res <- suppStat(response = trueZ[K + 1, -1], predictor = rowMeans(zAvg) )[3]
        resS <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3]

        resMat[iterNum, 189:192] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 193:196] <- colMeans(supMatS)  #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 197] <- res
        resMat[iterNum, 198] <- resS
        resMat[iterNum, 199] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 200] <- MSparams$lambda_z # tuning parameter
        resMat[iterNum, 201] <- MSparams$rho # cardinality of support
        resMat[iterNum, 202] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 213] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 227] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, supMat, supMatS, res, resS, betasMS, zStack, zAvg, warmStart)
        
        ########################################################
        # ******************************************************
        ####################################
        ####################################
        # MS4 -- different support with beta - betaBar penalty AND NO frobenius norm AND NO z- zbar penalty
        ####################################
        # share info on the beta - betaBar but no ||z - zbar|| penalty (currently but could do it if changed tuning grid)
        glPenalty <- 4
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        
        # tune multi-study with l0 penalty with GL Penalty = TRUE
        
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        res <- resS <- vector(length = K) # store support prediction
        
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
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaBstar <- MSparams$lambda2
            
            rhoG <- (rhoStar - 1):(rhoStar + 1)
            lambdaBgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaBstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaBgrid <- sort(lambdaBgrid, decreasing = TRUE)
            
            gridUpdate <- as.data.frame(  expand.grid( 0, lambdaBgrid, 0, rhoG) )
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
                                       messageInd = FALSE,
                                       LSitr = LSitr, 
                                       LSspc = LSspc,
                                       threads = tuneThreads,
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
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
        }
        
        MSparams <- tuneMS$best # tuned parameters
        
        # warm start
        warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                            y = as.vector(full$Y),
                            rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                            study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                            beta = betas,
                            lambda1 = MSparams$lambda1,
                            lambda2 = 0, #MSparams$lambda2,
                            lambda_z = MSparams$lambda_z,
                            scale = TRUE,
                            maxIter = 10000,
                            localIter = 50,
                            WSmethod = WSmethod,
                            ASpass = ASpass
        )
        
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
                          ASpass = ASpass
        )
        
        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS
        
        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)
        
        w <- coef(fitW)
        rm(fitW)
        
        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions
        
        resMat[iterNum, 66] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 67] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)
        
        # support stats
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking
        
        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j])
            supMatS[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack)
        }
        
        res <- suppStat(response = trueZ[K + 1, -1], predictor = rowMeans(zAvg) )[3]
        resS <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3]
        
        resMat[iterNum, 68:71] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 72:75] <- colMeans(supMatS)  #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 152] <- res
        resMat[iterNum, 153] <- resS
        resMat[iterNum, 99] <- MSparams$lambda2 # tuning parameter
        resMat[iterNum, 175] <- MSparams$rho # cardinality of support
        resMat[iterNum, 176] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 214] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 228] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, supMat, supMatS, res, resS, betasMS, zStack, zAvg, warmStart)
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
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        res <- resS <- vector(length = K) # store support prediction

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
                               threads = tuneThreads
        )

        MSparams <- tuneMS$best # parameters

        # warm start
        warmStart = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                        y = as.vector(full$Y),
                        rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = betas,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 50
        )

        # final model
        betasMS = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                        y = as.vector(full$Y),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = warmStart,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 50
        )

        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 76] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 77] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # support stats
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( betasMS[-1,] != 0) * 1 # stacking

        supMat <- supMatS <- matrix(NA, nrow = K, ncol = 4)
        for(j in 1:K){
            supMat[j,] <- suppStat(response = trueZ[j, -1], predictor = zAvg[, j] )
            supMatS[j,] <- suppStat(response = trueZ[j, -1], predictor = zStack )
        }

        res <- suppStat(response = trueZ[K + 1, -1], predictor = rowMeans(zAvg) )[3]
        resS <- suppStat(response = trueZ[K + 1, -1], predictor = zStack)[3]

        resMat[iterNum, 78:81] <- colMeans(supMat) #suppStat(response = z, predictor = zAvg * 1 ) # auc average weights ose
        resMat[iterNum, 82:85] <- colMeans(supMatS) #suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 154] <- res
        resMat[iterNum, 155] <- resS
        resMat[iterNum, 100] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 177] <- MSparams$rho # cardinality of support
        resMat[iterNum, 178] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 215] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 229] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, supMat, supMatS, res, resS, betasMS, zStack, zAvg, warmStart)
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
        tune.grid2$rho <- ncol(full) - 2 # full support
        tune.grid2 <- unique(tune.grid2)
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set

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
                               threads = tuneThreads
        )

        MSparams <- tuneMS$best # parameters

        betasMS = L0_MS(X = as.matrix( full[ , -c(1,2) ]) ,
                        y = as.vector(full$Y),
                        rho = MSparams$rho,
                        study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                        beta = betas,
                        lambda = MSparams$lambda,
                        scale = TRUE,
                        maxIter = 10000,
                        localIter = 0 # convex
        )

        # stack matrix
        predsMat <- cbind( 1, as.matrix( full[,-c(1,2) ] ) ) %*% betasMS
        # study by study predictions on test set
        testMat <- as.matrix( cbind(1, test[,-1] ) ) %*% betasMS

        # stacking -- nnls
        fitW <- glmnet(y = as.vector(full$Y),
                       x = as.matrix(predsMat),
                       alpha = 0,
                       lambda = 0,
                       standardize = TRUE,
                       intercept = TRUE,
                       thresh = 1e-10,
                       lower.limits = 0)

        w <- coef(fitW)
        rm(fitW)

        stackPreds <- cbind(1, testMat ) %*% w # stacking predictions
        avgPreds <- rowMeans( testMat ) # average predictions

        resMat[iterNum, 86] <- sqrt( mean( (test$Y - avgPreds )^2 ) ) # RMSE of ose Avg weights
        resMat[iterNum, 87] <- sqrt( mean( (test$Y - stackPreds)^2 ) ) # RMSE ose stacking weights
        rm(stackPreds, predsMat, avgPreds, testMat)

        # stacking auc
        zStack <- I( betasMS %*% w[-1] != 0)[-1] * 1 # stacking
        zAvg <- I( rowMeans(betasMS)[-1] != 0) * 1 # stacking

        resMat[iterNum, 88:91] <- suppStat(response = z, predictor = zAvg ) # auc average weights ose
        resMat[iterNum, 92:95] <- suppStat(response = z, predictor = zStack ) # auc stacking
        resMat[iterNum, 101] <- MSparams$lambda # tuning parameter
        resMat[iterNum, 179] <- MSparams$rho # cardinality of support
        resMat[iterNum, 180] <- sum( zStack ) # cardinality of support
        resMat[iterNum, 216] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
        resMat[iterNum, 230] <- multiTaskRmse(data = mtTest, beta = betasMS)
        
        rm(w, MSparams, tuneMS, zAvg, zStack)
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
