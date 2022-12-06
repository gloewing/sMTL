# uses active set versions appropriate for each method
library(pROC)
library(JuliaConnectoR)

library(caret)
library(glmnet)
library(dplyr)
library(L0Learn)

source("SimFn.R")

sims6 <- read.csv("sparseParam_test")
cluserInd <- TRUE # whether running on computer or on cluster

save.folder <- "/n/home12/gloewinger/sparseMT6"
load.folder <- "~/Desktop/Research"
if(cluserInd){
    # if on cluster
    args = commandArgs(TRUE)
    runNum <- as.integer( as.numeric(args[1]) ) 
    iterNum <- as.integer( Sys.getenv('SLURM_ARRAY_TASK_ID') ) # seed index from array id

    # Julia paths
    juliaPath <- "/n/sw/eb/apps/centos7/Julia/1.5.3-linux-x86_64/bin"
    juliaFnPath_MT <- juliaFnPath <- "/n/home12/gloewinger/"
    
}else{
    runNum <- 1
    setwd("~/Desktop/Research")
    iterNum <- 1

    # Julia paths
    juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
    juliaFnPath_MT <- juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune MT/"

}

Sys.setenv(JULIA_BINDIR = juliaPath)

# sim params
simNum <- runNum
totalSims <- 100
p2 <- FALSE # p2 or p5

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
 
categor <- 4 #FALSE # if true: cardinality of random support is fixed, if FALSEE: random (iid Bernouli) 
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
covType <- sims6$cov[runNum] # type of covariance matrix for the features
rho_corr <- sims6$rho[runNum] # rho used in covariance matrix for features if "exponential" or "pairwaise"
MoM <- FALSE # indicator of whether to run MoM
tuneInterval <- 10 # divide/multiple optimal value by this constant when updating tuning
gridLength <- 10 # number of values between min and max of grid constructed by iterative tuning
LSitr <- 50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- 1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 50 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
maxIter_cv <- 5000

errorMult <- sims6$errorMult[runNum] # range of error for uniform
tau <- sims6$tau[runNum]
epsHigh <- tau * errorMult# noise lower/upper
epsLow <- tau / errorMult# noise lower/upper
nLow <- nHigh <- sims6$nLow[runNum]  # multiply by 2 because of multi-task test set  # samp size lower/upper
tuneInd <- sims6$tuneInd[runNum]
# nHigh <- sims6$nLow[runNum] * errorMult # multiply by 2 because of multi-task test set  # samp size lower/upper
# nLow <- sims6$nLow[runNum] / errorMult
WSmethod = 2 # sims6$WSmethod[runNum] ---- 2
ASpass = TRUE # sims6$ASpass[runNum] ----- TRUE # next

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
MSTn <- sims6$multiTask[runNum] #"hoso" #"balancedCV" # tuning for MS (could be "hoso")
nfold <- 10

lamb <- 0.5
fileNm <- paste0("sprsMS_LS_algos_",
                 "_p2_", p2,
                "_s_", s, "_r_", r, "_rp_", r_p,
                "_q_", q,
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
               "_TnIn_", tuneInd,
               "cat_", categor)

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
    objCalc <- juliaCall("include", paste0(juliaFnPath_MT, "objCalc_MT.jl") ) 
    ####################################################

    totalMethods <- 6
    resMat <- matrix(nrow = totalSims, ncol = totalMethods * 4)
    colnames(resMat) <- c( paste0( c("time_", "mse_", "coef_", "obj_" ), rep(1:totalMethods, each = 4) )
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
            q <- max(r_card, q) # make sure it is big enough to select at least r_card
            
            m <- max(sSeq) + 2
            suppSeq <- seq(m, m + 2 * (q - 1), by = 2) #seq(2*s + 3, 2*(s + r) + 1, by = 2) # alternating sequence of covariate indices starting after common support that is "r" long
            
            for(j in 1:totalStudies){
                # fixed cardinality: categorical random variables
                suppRandom <- sample(suppSeq, r_card, replace = FALSE)
                fixB[j, suppRandom ] <- 1 # only add the ones that are not zeroed out
            }
            
            # cardinality of support
            card <- sum( fixB[1,] ) # all have the same cardinaity so choose first task arbitrarily
            
            # 3 above and below true support
            minRho <- max(1,   card - 3   )
            maxRho <- min(   card + 3  )
            
        }
        # Z <- matrix(0, nrow = p, ncol = K)
        
        # overlap of true support
        #resMat[iterNum, 129] <- suppressWarnings( suppHet(t(fixB), intercept = FALSE)[1] )
        
        
        print(fileNm)

        ##########################################
        rho <- card # just use this to speed up tuning for obj experiments
        lambda <- sort( unique( c(0, 1e-5, 1e-6, 5,10, 50, 100, 200,
                                  exp(-seq(0,9, length = 50))
        ) ), decreasing = TRUE ) #
        
        lambdaShort <- sort( unique( c(0,
                                       exp(-seq(0,5, length = 5)),
                                       5,10, 50, 100, 250, 500, 1000, 2500, 5000, 10000) ),
                             
                             decreasing = TRUE ) 
        lambdaZ <- sort( unique( c(0, 1e-6, 1e-5, 1e-4, 1e-3,
                                   exp(-seq(0,5, length = 8)),
                                   1:3) ),
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
                               corr_rho = rho_corr # used if pariwise or exponential correlation
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

        ####################################
        #  L0 regularization with ||z - zbar|| penalty and (potentially) ||beta - betaBar|| and (potentially) frobenius norm
        # glPenalty = 2
        ####################################
        timeStart1 <- Sys.time()
        
        predsMat <- matrix(NA, ncol = K, nrow = nrow(full)) # predictions for stacking matrix
        # testMat <- matrix(NA, ncol = K, nrow = nrow(test)) # predictions on test set
        res <- resS <- vector(length = K) # store support prediction
        
        if(p2){
            
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
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # original from 1/20/21
            # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
            # ************
            gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoStar) )
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
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
        }else{
            
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
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
            MSparams <- tuneMS$best # parameters
            rhoStar <- MSparams$rho
            lambdaZstar <- MSparams$lambda_z
            
            rhoG <- rhoStar # for speed keep as just correct rho
            lambdaZgrid<- c( seq(3, 10, length = 5), seq(0.5, 2, length = 10), seq(0.1, 1, length = 5) ) * lambdaZstar # makes grid roughly spaced between   # exp(-seq(0, 2.3, length = 5))
            lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)
            
            # ************
            # original from 1/20/21
            # gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoG) )
            # ************
            gridUpdate <- as.data.frame(  expand.grid( 0, lambda, lambdaZgrid, rhoG) )
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
                                       maxIter = maxIter_cv,
                                       threads = tuneThreads,
                                       WSmethod = WSmethod,
                                       ASpass = ASpass
            )
            
        }
 
        
        MSparams <- tuneMS$best # parameters
        rm(tuneMS)
        
        b <- matrix(0, ncol = K, nrow = p + 1)
        
        set.seed(iterNum)
        itrOrder <- sample.int(totalMethods, totalMethods, replace = FALSE) # randomize order
        for(itr in itrOrder){
            
            if(itr == 1){
                
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                WSmethod = 2 # sims6$WSmethod[runNum] ---- 2
                ASpass = TRUE # sims6$ASpass[runNum] ----- TRUE # next
                
                
            }else if(itr == 2){
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_tuneTest_MT.jl") ) # MT: Need to check it works; no active set but NO common support (it does have Z - zbar and beta - betabar)
                # ran bad
                
            }else if(itr == 3){
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study
                
            }else if(itr == 4){
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                WSmethod = 1 # sims6$WSmethod[runNum] ---- 2
                ASpass = TRUE # sims6$ASpass[runNum] ----- TRUE # next
                
            }else if(itr == 5){
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                WSmethod = 2 # sims6$WSmethod[runNum] ---- 2
                ASpass = FALSE # sims6$ASpass[runNum] ----- TRUE # next
                
            }else if(itr == 6){
                rm(L0_MS_z)
                L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty
                WSmethod = 1 # sims6$WSmethod[runNum] ---- 2
                ASpass = FALSE # sims6$ASpass[runNum] ----- TRUE # next
                
            }
            
            timeStart1 <- timeStart <- Sys.time()
            
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
                              study = NA, # these are the study labels ordered appropriately for this fold
                              beta = warmStart,
                              lambda1 = MSparams$lambda1,
                              lambda2 = MSparams$lambda2,
                              lambda_z = MSparams$lambda_z,
                              scale = TRUE,
                              maxIter = 10000,
                              localIter = 0,#localIters, #localIters,
                              WSmethod = WSmethod,
                              ASpass = ASpass
            )
            
            timeEnd1 <- timeEnd <- Sys.time()
            
            resMat[iterNum, (itr - 1) * 4 + 1] <- as.numeric(difftime(timeEnd1, timeStart1, units='mins'))
            
            resMat[iterNum, (itr - 1) * 4 + 4] <- objCalc(X = as.matrix( full[ , Xindx ]) ,
                                                          y = as.matrix( full[, Yindx] ),
                                                          beta = betasMS,
                                                          lambda1 = MSparams$lambda1,
                                                          lambda2 = MSparams$lambda2,
                                                          lambda_z = MSparams$lambda_z)
            
            resMat[iterNum, (itr - 1) * 4 + 3] <- sqrt( mean( (betasMS - trueB)^2 ) ) # coef error
            resMat[iterNum, (itr - 1) * 4 + 2] <- multiTaskRmse_MT(data = mtTest, beta = betasMS)
            print(difftime(timeEnd, timeStart, units='mins'))
            
            rm(betasMS, warmStart )
            
        }

        
        ########################################################
       
        ########################
        # save results
        ########################
        print("setWD to save file")
        saveFn(file = resMat, 
               fileNm = fileNm, 
               iterNum = iterNum, 
               save.folder = save.folder)
        
        #####################################################################
