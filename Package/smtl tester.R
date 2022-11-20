library(sMTL)
smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", installJulia = FALSE, installPackages = FALSE)


######################
# L0 Regression Single
######################
X <- matrix(rnorm(1000), ncol = 10)

beta <- c(1, 1, rep(0, ncol(X) - 2) )

y <- X %*% beta + rnorm(nrow(X))

study <- sample.int(5, nrow(X), replace=TRUE)

beta = sMTL::smtl(y = as.numeric(y), 
                X = X, 
                study = NA,#study, 
                s = 3, 
                commonSupp = TRUE,
                warmStart = TRUE,
                # multiTask = TRUE,
                lambda_1 = c(0.1, 0.001), 
                lambda_2 = c(0,0), 
                lambda_z = c(0,0), 
                scale = TRUE,
                maxIter = 5000,
                LocSrch_maxIter = as.integer(50)
)


###########################


sims6 <- read.csv("~/Desktop/Research/sparseParam_test")
source("~/Desktop/Research/SimFn.R")

cluserInd <- TRUE # whether running on computer or on cluster

runNum <- 1
iterNum <- 1
 
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
s <- 5#sims6$s[runNum] #
r <- 5 #sims6$r[runNum]
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
LocSrch_maxIter <- 50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LocSrch_skip <- 1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LocSrch_skip> number of tuning parameters (like every fifth value)
LocSrch_maxIters <- 50 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning

errorMult <- sims6$errorMult[runNum] # range of error for uniform
tau <- sims6$tau[runNum]
epsHigh <- tau * errorMult# noise lower/upper
epsLow <- tau / errorMult# noise lower/upper
nLow <- nHigh <- sims6$nLow[runNum]  # multiply by 2 because of multi-task test set  # samp size lower/upper
tuneInd <- sims6$tuneInd[runNum]
# nHigh <- sims6$nLow[runNum] * errorMult # multiply by 2 because of multi-task test set  # samp size lower/upper
# nLow <- sims6$nLow[runNum] / errorMult
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

#oseTuneLong <- sims6$oseTn[runNum]

#nfold <- ifelse(nfold == "K", K, nfold)
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
    c(lambda) , # 0 # add 0 but not to glmnet because that will cause problems
    rho)
) # tuning parameters to consider
colnames(tune.grid) <- c("lambda", "rho")

# glmnet for ridge
tune.gridGLM <- as.data.frame( cbind(0, lambda) ) # Ridge
colnames(tune.gridGLM) <- c("alpha", "lambda")

timeStart <- Sys.time()

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

full <- as.data.frame( full$data )

library(sMTL)
smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", 
           installJulia = FALSE, 
           installPackages = FALSE)

study <- sample.int(5, nrow(full), replace=TRUE)
######################
# multi-study
######################

grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = 0, 
                   lambda_z = c(0.01, 0.1,1,5))

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

grid2 <- grid
grid2$s <- ncol(full) - 1
tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid2,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 


tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = as.integer(study), 
                  s = 251, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0,0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

preds <- predict(model = beta, 
                 X = as.matrix( full[,-2] ), 
                 lambda_1 = 0.01, 
                 lambda_2 = 0, 
                 lambda_z = 0,
                 stack = TRUE)

######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = c(0.01, 0.1,1,5), lambda_z = c(0.01, 0.1,1,5))

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

grid2 = grid
grid2$s=251
tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid2,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid2,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 


tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid2,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = as.integer(study), 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

preds <- predict(model = beta, 
                 X = as.matrix( full[,-2] ), 
                 lambda_1 = 0.01, 
                 lambda_2 = 0.1, 
                 lambda_z = 0,
                 stack = FALSE)

######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = c(0.01, 0.1,1,5), lambda_z = c(0.01, 0.1,1,5))

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

grid2 <- grid
grid2$s <- ncol(full) - 1
tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 


beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = as.integer(study), 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0,0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = as.integer(study), 
                  s = 251, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0,0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)


######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = c(0.01, 0.1,1,5), lambda_z = c(0.01, 0.1,1,5))

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = as.integer(study), 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 


beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = as.integer(study), 
                      s = 10, 
                      commonSupp = FALSE,
                      warmStart = TRUE,
                      # multiTask = TRUE,
                      lambda_1 = c(0.01, 0.1), 
                      lambda_2 = c(0.1,0.1), 
                      lambda_z = c(0.1,0.1), 
                      scale = TRUE,
                      maxIter = 5000,
                      LocSrch_maxIter = 50
)

######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = c(0.01, 0.1,1,5), lambda_z = c(0.01, 0.1,1,5))

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = c(0.1,0.1), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50
)


######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = 0, lambda_z = 0)

tn <- sMTL::cv.smtl(y = as.matrix(full[,1]), 
                    X = as.matrix( full[,-seq(1,2)] ), 
                    study = study, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 


tn <- sMTL::cv.smtl(y = as.matrix(full[,1]), 
                    X = as.matrix( full[,-seq(1,2)] ), 
                    study = study, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0, 0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50
)


######################

grid <- data.frame(s = c(2,2,3,3), lambda_1 = c(0.01, 0.1,1,5), lambda_2 = 0, lambda_z = 0)

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0, 0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50,
                  independent.regs = FALSE
)



######################
# multi-task
######################
grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = c(0.01, 0.1,1,5), 
                   lambda_z = 0
                   )

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = 0, 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0,0), 
                  lambda_z = 0, 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)



######################
grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = c(0.01, 0.1,1,5), 
                   lambda_z = 0.1
)

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = c(0.1,0.1), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50
)

######################

grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = c(0.01, 0.1,1,5), 
                   lambda_z = 0.1
)

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = c(0.1,0.1), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50
)

######################

grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = 0, 
                   lambda_z = 0
)

tn <- sMTL::cv.smtl(y = as.matrix(full[,1:5]), 
                    X = as.matrix( full[,-seq(1,5)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE,
                    independent.regs = TRUE # shared active sets
) 

beta = sMTL::smtl(y = as.matrix(full[,1:5]), 
                  X = as.matrix( full[,-seq(1,5)] ), 
                  study = NA, 
                  s = 2, 
                  commonSupp = FALSE,
                  warmStart = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0, 0), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = as.integer(5000),
                  LocSrch_maxIter = 50,
                  independent.regs = TRUE # shared active sets
)

preds <- predict(model = beta, 
                   X = as.matrix( full[,-seq(1,5)] ), 
                   lambda_1 = 0.01, 
                   lambda_2 = 0, 
                   lambda_z = 0,
                   stack = FALSE)

# test out stacking
beta$y_train <- beta$y_train[,1]
preds <- predict(model = beta, 
                 X = as.matrix( full[,-seq(1,5)] ), 
                 lambda_1 = 0.01, 
                 lambda_2 = 0, 
                 lambda_z = 0,
                 stack = TRUE)



######################

grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = 0, 
                   lambda_z = 0
)

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-c(2,3)] ), 
                    study = NA, 
                    commonSupp = FALSE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE,
                    independent.regs = FALSE # shared active sets
) 


beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01, 0.1), 
                  lambda_2 = c(0.1,0.1), 
                  lambda_z = c(0, 0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

preds <- predict(model = beta, 
                 X = as.matrix( full[,-2] ), 
                 lambda_1 = 0.01, 
                 lambda_2 = 0.1, 
                 lambda_z = 0,
                 stack = FALSE)


######################

grid <- data.frame(s = c(2,2,3,3), 
                   lambda_1 = c(0.01, 0.1,1,5), 
                   lambda_2 = 0, 
                   lambda_z = 0
)

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = NA, 
                    commonSupp = TRUE,
                    grid = grid,
                    nfolds = 5,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE,
                    independent.regs = TRUE # shared active sets
) 

beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = TRUE,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01), 
                  lambda_2 = c(0.1), 
                  lambda_z = c(0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50
)

preds <- predict(model = beta, 
                 X = as.matrix( full[,-2] ), 
                 lambda_1 = NA, 
                 lambda_2 = NA, 
                 lambda_z = NA,
                 stack = FALSE)


# grid <- data.frame(s = 2, lambda_1 = c(0.01, 0.1,1), lambda_2 = c(0.01, 0.1,1), lambda_z = c(0.01, 0.1,1))
grid <- data.frame(s = 2, lambda_1 = c(0.01, 0.1,1), lambda_2 = 0, lambda_z = 0)

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
            X = as.matrix( full[,-2] ), 
            study = NA, 
            grid = grid,
            nfolds = 5,
            commonSupp = FALSE,
            multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
            maxIter = 1000,
            LocSrch_skip = 1,
            LocSrch_maxIter = 50,
            messageInd = TRUE
    ) 


grid <- data.frame(s = 2, lambda_1 = c(0.01, 0.1,1), lambda_2 = c(0.01, 0.1,1), lambda_z = c(0.01, 0.1,1))
study <- sample.int(5, nrow(full), replace=TRUE)

tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = study, 
                    grid = grid,
                    nfolds = 5,
                    commonSupp = FALSE,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 



tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = study, 
                    grid = grid,
                    nfolds = 5,
                    commonSupp = FALSE,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE
) 

grid$lambda_2=0
grid$lambda_z=0


tn <- sMTL::cv.smtl(y = as.numeric(full[,2]), 
                    X = as.matrix( full[,-2] ), 
                    study = study, 
                    grid = grid,
                    nfolds = 5,
                    commonSupp = FALSE,
                    multiTask = FALSE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE,
                    independent.regs = TRUE
) 


beta = sMTL::smtl(y = as.numeric(full[,2]), 
                  X = as.matrix( full[,-2] ), 
                  study = study, 
                  s = 10, 
                  commonSupp = F,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01), 
                  lambda_2 = c(0), 
                  lambda_z = c(0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50,
                  messageInd = TRUE,
                  independent.regs = TRUE
)


## multi-label separate active sets

tn <- sMTL::cv.smtl(y = as.matrix(full[,2:5]), 
                    X = as.matrix( full[,-seq(2,5)] ), 
                    study = study, 
                    grid = grid,
                    nfolds = 5,
                    commonSupp = FALSE,
                    multiTask = TRUE, # if TRUE and study indices are provided then do a merged LO regression tuned with hoso
                    maxIter = 1000,
                    LocSrch_skip = 1,
                    LocSrch_maxIter = 50,
                    messageInd = TRUE,
                    independent.regs = TRUE
) 


beta = sMTL::smtl(y = as.matrix(full[,2:5]), 
                  X = as.matrix( full[,-seq(2,5)] ), 
                  study = NA, 
                  s = 10, 
                  commonSupp = F,
                  warmStart = TRUE,
                  # multiTask = TRUE,
                  lambda_1 = c(0.01), 
                  lambda_2 = c(0), 
                  lambda_z = c(0), 
                  scale = TRUE,
                  maxIter = 5000,
                  LocSrch_maxIter = 50,
                  messageInd = TRUE,
                  independent.regs = TRUE
)

