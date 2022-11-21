library(MASS)
library(JuliaConnectoR)
library(ggpubr)

setwd("~/Desktop/Research")

# Figures and Tables
f1score <- function(data, name){
    tp <- data[ paste0(name, "_tp") ]
    fp <- data[ paste0(name, "_fp") ]
    
    return( tp / (tp + 0.5 * (fp + 1 - tp ) ) ) # f1 score
}

LSitr <- 50 #50 #5 #ifelse(is.null(sims6$lit)[runNum], 50, sims6$lit[runNum] ) # number of iterations of local search to do while tuning (for iterations where we do actually use local search)
LSspc <- 1 #1#1 #5 #ifelse(is.null(sims6$lspc)[runNum], 1, sims6$lspc[runNum] ) # when tuning, do local search every <LSspc> number of tuning parameters (like every fifth value)
localIters <- 50 # 0 # number of LS iterations for actually fitting models
tuneThreads <- 1 # number of threads to use for tuning
lambdaZmax <- 5 # make sure lambda_z are smaller than this to pervent numerical instability
maxIter_train <- 5000
maxIter_cv <- 1000

tuneInd <- TRUE
WSmethod <- 2 
ASpass <- TRUE 
MSTn <- "multiTask"

################
# Simulate Data
################
set.seed(1)
num.studies <- K <- 2
nfold <- 5
n <- 25 # n = 50
numCovs <- p <- 50
s <- 0
r <- 14 # r = 10
r_card <- 2 * r
r_p <- 0.5 # probability of inclusion
corr_rho <- 0.5
sigma2 <- 1 # residual variance
zeroCovs <- c() #seq(2, numCovs + 1)[-seq(2, 2 * s, by = 2)] # alternate because of exponential correlation structure

Sigma <- matrix(1, ncol = numCovs, nrow = numCovs)

# exponential correlation
for(i in 1:numCovs){
    for(j in 1:numCovs){
        Sigma[i,j] <- corr_rho^( abs(i - j) )
    }
}

# design matrix
X <-  mvrnorm(n,
              mu = rep(0, times = p  ),
              Sigma = Sigma)

# support matrix
Z <- matrix(0, nrow = p, ncol = K)

suppSeq <- seq(2, 4 * round(r * r_p) + 1, by = 2) #seq(2*s + 3, 2*(s + r) + 1, by = 2) # alternating sequence of covariate indices starting after common support that is "r" long

for(j in 1:K){
    
    # fixed cardinality: categorical random variables
    r_card <- round(r * r_p) # cardinality: total number of 1s is roughly same in expectation
    suppRandom <- sample(suppSeq, r_card, replace = FALSE)
    
    Z[suppRandom, j ] <- 1 # only add the ones that are not zeroed out
}

# beta matrix
B <- matrix(0, nrow = p + 1, ncol = K)
B[,1] <- 1 # first task has coefs of 1
B[,2] <- -1 # second task has coefs of -1
B <- B[-1,] * Z
rho <- sum(Z) / K
    
# outcome matrix
Y <- X %*% B + rnorm(n, mean = 0, sd = sqrt(sigma2) )
full <- cbind(Y, X)
colnames(full) <- c( "Y_1", "Y_2", paste0("X_", 1:p)  )
Yindx <- 1:K
Xindx <- seq(1, ncol(full))[-Yindx]

# tuning parameters
lambda <- sort( unique( c(1e-6, 1e-5, 0.0001, 0.001, 0.01, 5,10, 50, 100,
                          exp(-seq(0,5, length = 30))
                            ) ), 
                decreasing = TRUE 
                ) 

lambdaZ <- sort( unique( c(0, c( 10^c(-(6:3)), 5 * 10^c(-(6:3))),
                           exp(-seq(0,5, length = 8)),
                           1:3) ),
                 decreasing = TRUE ) # 2:100

# Julia paths
source("sparseFn_iht_test_MT.R") # USE TEST VERSION HERE
sparseCV_iht_par <- sparseCV_iht

juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
juliaFnPath_MT <- juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune MT/"
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


############################
# GLMNET Lasso - MultiTask
############################
library(glmnet)
tune.mod <- cv.glmnet(y = as.matrix(full[,Yindx]),
                      x = as.matrix(full[,Xindx]),
                      alpha = 1,
                      intercept = TRUE,
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

# best lasso with s rho constraint
betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = lambdaStar) ) )
betaEst <- do.call(cbind, as.list( coef(tune.mod, exact = TRUE, s = "lambda.min") ) )

gLasso <- as.matrix(betaEst)

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

rm(L0_MS_z3)
L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexact_diffAS_tuneTest_MT.jl") ) # sepratae active sets for each study

# warm start
warmStart = L0_MS_z3(X = as.matrix( full[ , Xindx ]) ,
                     y = as.matrix( full[, Yindx] ),
                     rho = min(L0_tune$rho * 4, numCovs -1),
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


oseL0 <- betas

###################
# Zbar + L2
###################
tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambda, 0, lambdaZ, rho) )
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
lambdaZstar <- MSparams$lambda_z

lambdaZgrid <- c( seq(1.5, 10, length = 5), seq(0.1, 1, length = 5) ) * lambdaZstar
lambdaZgrid <- lambdaZgrid[lambdaZgrid <= lambdaZmax] # make sure this is below a threshold to prevent numerical issues
lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)


gridUpdate <- as.data.frame(  expand.grid( lambda, 0, lambdaZgrid, rhoStar) )
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
                           messageInd = TRUE,
                           LSitr = LSitr, 
                           LSspc = LSspc,
                           maxIter = maxIter_cv,
                           threads = tuneThreads,
                           WSmethod = WSmethod,
                           ASpass = ASpass
)

MSparams <- tuneMS$best # parameters

rm(L0_MS_z)
L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty

# warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                    y = as.matrix( full[, Yindx] ),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
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

# final model # optimal lambda = 0.05744622
beta_zbar = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                  y = as.matrix( full[, Yindx] ),
                  rho = MSparams$rho,
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

################
# lambda_z= 0
################
beta_lamba0 = oseL0

#################
# lambda = 0.05
#################
lamb = 0.05
b <- matrix(0, ncol = K, nrow = numCovs + 1)

tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambda, 0, lamb, rho) )
colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z", "rho")

# order correctly
tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(tune.grid_MSZ_5$rho,
                                           tune.grid_MSZ_5$lambda1,
                                           -tune.grid_MSZ_5$lambda_z,
                                           decreasing=TRUE),     ]
# tune
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
                           maxIter = maxIter_cv,
                           WSmethod = WSmethod,
                           ASpass = ASpass
)

# warm start 
warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                    y = as.matrix( full[, Yindx] ),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                    beta = b,
                    lambda1 = max(MSparams$lambda1, 1e-6), # ensure theres some regularization given higher rho for WS
                    lambda2 = 0, #MSparams$lambda2,
                    lambda_z = 0,
                    scale = TRUE,
                    maxIter = maxIter_train,
                    localIter = 0,
                    WSmethod = WSmethod,
                    ASpass = ASpass)


beta_lamba01 = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                      y = as.matrix( full[, Yindx] ),
                      rho = MSparams$rho,
                      beta = warmStart,
                      lambda1 = MSparams$lambda1,
                      lambda2 = MSparams$lambda2,
                      lambda_z = lamb,
                      scale = TRUE,
                      maxIter = maxIter_train,
                      localIter = localIters,
                      WSmethod = WSmethod,
                      ASpass = ASpass
)

# lambda = 1
#################
# lambda = 0.005
#################

tune.grid_MSZ_5 <- as.data.frame(  expand.grid( lambda, 0, 1, rho) )
colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z", "rho")

# order correctly
tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(tune.grid_MSZ_5$rho,
                                           tune.grid_MSZ_5$lambda1,
                                           -tune.grid_MSZ_5$lambda_z,
                                           decreasing=TRUE),     ]

# tune
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
                           maxIter = maxIter_cv,
                           WSmethod = WSmethod,
                           ASpass = ASpass
)

# warm start 
warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                    y = as.matrix( full[, Yindx] ),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                    beta = b,
                    lambda1 = max(MSparams$lambda1, 1e-6), # ensure theres some regularization given higher rho for WS
                    lambda2 = 0, #MSparams$lambda2,
                    lambda_z = 0,
                    scale = TRUE,
                    maxIter = maxIter_train,
                    localIter = 0,
                    WSmethod = WSmethod,
                    ASpass = ASpass)

beta_lamba1 = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                       y = as.matrix( full[, Yindx] ),
                       rho = MSparams$rho,
                       beta = warmStart,
                       lambda1 = MSparams$lambda1,
                       lambda2 = MSparams$lambda2,
                       lambda_z = 1,
                       scale = TRUE,
                       maxIter = maxIter_train,
                       localIter = localIters,
                       WSmethod = WSmethod,
                       ASpass = ASpass
)

##########
# Bbar
##########

tune.grid_beta <- as.data.frame(  expand.grid( 0, lambda, 0, rho) ) # tuning parameters to consider
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
  
MSparams <- tuneMS$best # tuned parameters

rm(L0_MS_z)
L0_MS_z <- juliaCall("include", paste0(juliaFnPath_MT, "BlockComIHT_inexactAS_tune_old_MT.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty

# warm start
warmStart = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                    y = as.matrix( full[, Yindx] ),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
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
bbar = L0_MS_z(X = as.matrix( full[ , Xindx ]) ,
                  y = as.matrix( full[, Yindx] ),
                  rho = MSparams$rho,
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


########################
# common support
########################

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
                  beta = b,
                  lambda = MSparams$lambda,
                  scale = TRUE,
                  maxIter = maxIter_train,
                  localIter = 0
)

# final model
beta_cs = L0_MS(X = as.matrix( full[ , Xindx ]) ,
                y = as.matrix( full[, Yindx] ),
                rho = MSparams$rho,
                beta = warmStart,
                lambda = MSparams$lambda,
                scale = TRUE,
                maxIter = maxIter_train,
                localIter = localIters
                )


betas <- list("CS+L2" = beta_cs,
              "Zbar+L2" = beta_zbar,
              "Bbar" = bbar,
              "TS-SR" = oseL0,
              "LASSO" = gLasso,
              "Betas" = B
              )

saveRDS(betas, 
        "~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure/betas")


betas <- readRDS("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure/betas")

# plots
library(tidyverse)
library(ggplot2)
library(latex2exp)
library(grid)
library(ggpubr)


betas$Betas <- rbind(0, betas$Betas) # add on intercept to be consistent with estimates (removed below)
for(j in 1:length(betas)){
    
    # vectorize
    betas[[j]] <- data.frame(
                        cbind( as.factor(rep(1:K, each = p)),
                        as.vector( betas[[j]][-1,] )
                        )
    )
    
    # add names
    betas[[j]] <- cbind( rep(1:p, K), names(betas)[j], betas[[j]] )
    colnames( betas[[j]] ) <- c("Index", "Method", "Task", "Beta")
}

b <- do.call(rbind, betas)
b$Task <- as.factor(b$Task)
# 
# p1 <- ggplot(b, aes(y = Beta, x = Index)) + geom_line() + theme_minimal() + 
#     theme(axis.title.x = element_blank(), axis.text.x = element_blank())


for(j in 1:length(unique(b$Method)) ){
    
    method = unique(b$Method)[j]
    
    #Plot
    if(j == length(unique(b$Method)) ){
        # true betas -- uses different title
        assign(paste0("p", j),
               b %>% as_tibble() %>%
                   filter(Method == method) %>%
                   ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
                   geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
                   geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
                   theme_bw() +
                   ylab(TeX('True $\\mathbf{\\beta}$') ) + 
                   theme_classic() + 
                   theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # remove x-axis
                   geom_hline(yintercept = 0, linetype = "dashed") +
                   scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5) ) +
                   scale_color_manual(values=c("blue", "red")) +
                   theme(legend.position="none",
                    plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                    axis.text=element_text(face="bold",color="black", size=rel(1)),
                    axis.title = element_text(face="bold", color="black", size=rel(1)),
                    legend.key.size = unit(2, "line"), # added in to increase size
                    legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                    legend.title = element_text(face="bold", color="black", size = rel(1)),
                    strip.text.x = element_text(face="bold", color="black", size = rel(1))
                    )
                 )  
            
        
    }else if(j == 5  ){
      # make the legend at the bottom here because we use it as the last figure
      assign(paste0("p", j),
             b %>% as_tibble() %>%
               filter(Method == method) %>%
               ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
               # geom_point() +
               geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
               geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
               theme_bw() +
               ylab(TeX(paste0(method, ' $\\hat{\\mathbf{\\beta}}$') )) + 
               xlab("Coefficient Index") + 
               theme_classic() +
               geom_hline(yintercept = 0, linetype = "dashed") +
               scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5)) + 
               scale_x_continuous(breaks = c(1, 10, 20, 30, 40, 50)) + 
               scale_color_manual(values = c("blue", "red")) +
               theme(plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                      axis.text=element_text(face="bold",color="black", size=rel(1)),
                      axis.title = element_text(face="bold", color="black", size=rel(1)),
                      legend.key.size = unit(2, "line"), # added in to increase size
                      legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                      legend.title = element_text(face="bold", color="black", size = rel(1)),
                      strip.text.x = element_text(face="bold", color="black", size = rel(1)),
                      legend.position="bottom",
                      # legend.justification="right",
                      legend.margin=margin(0,0,0,0),
                      legend.box.margin=margin(-10,-10,-10,-10)
               )
      )
      
      
      
    }else if(j == 2  ){
      # make the legend at the bottom here because we use it as the last figure
      assign(paste0("p2", j),
             b %>% as_tibble() %>%
               filter(Method == method) %>%
               ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
               # geom_point() +
               geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
               geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
               theme_bw() +
               ylab(TeX(paste0(method, ' $\\hat{\\mathbf{\\beta}}$') )) + 
               xlab("Coefficient Index") + 
               theme_classic() +
               geom_hline(yintercept = 0, linetype = "dashed") +
               scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5)) + 
               scale_x_continuous(breaks = c(1, 10, 20, 30, 40, 50)) + 
               scale_color_manual(values = c("blue", "red")) +
               theme(plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                     axis.text=element_text(face="bold",color="black", size=rel(1)),
                     axis.title = element_text(face="bold", color="black", size=rel(1)),
                     legend.key.size = unit(2, "line"), # added in to increase size
                     legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                     legend.title = element_text(face="bold", color="black", size = rel(1)),
                     strip.text.x = element_text(face="bold", color="black", size = rel(1)),
                     legend.position="bottom",
                     # legend.justification="right",
                     legend.margin=margin(0,0,0,0),
                     legend.box.margin=margin(-10,-10,-10,-10)
               )
      )
      
      # make the legend at the bottom here because we use it as the last figure
      assign(paste0("p", j),
             b %>% as_tibble() %>%
               filter(Method == method) %>%
               ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
               geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
               geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
               theme_bw() +
               ylab(TeX('True $\\mathbf{\\beta}$') ) + 
               theme_classic() + 
               theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # remove x-axis
               geom_hline(yintercept = 0, linetype = "dashed") +
               scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5) ) +
               scale_color_manual(values=c("blue", "red")) +
               theme(legend.position="none",
                     plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                     axis.text=element_text(face="bold",color="black", size=rel(1)),
                     axis.title = element_text(face="bold", color="black", size=rel(1)),
                     legend.key.size = unit(2, "line"), # added in to increase size
                     legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                     legend.title = element_text(face="bold", color="black", size = rel(1)),
                     strip.text.x = element_text(face="bold", color="black", size = rel(1))
               )
      )  
      
      
      
    }else{
      # all other figures
        assign(paste0("p", j),
               b %>% as_tibble() %>%
                   filter(Method == method) %>%
                   ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
                   # geom_point() +
                   geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
                   geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
                   theme_bw() +
                   ylab(TeX(paste0(method, ' $\\hat{\\mathbf{\\beta}}$') )) + 
                   theme_classic() +
                   theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # remove x-axis
                   geom_hline(yintercept = 0, linetype = "dashed") +
                   scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5)) + 
                   scale_color_manual(values = c("blue", "red")) +
                   theme(legend.position="none",
                       plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                       axis.text=element_text(face="bold",color="black", size=rel(1)),
                       axis.title = element_text(face="bold", color="black", size=rel(1)),
                       legend.key.size = unit(2, "line"), # added in to increase size
                       legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                       legend.title = element_text(face="bold", color="black", size = rel(1)),
                       strip.text.x = element_text(face="bold", color="black", size = rel(1))
                 )
        )
        
    }
    
}

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_n25_r14_p25_s5.pdf",
        plot = grid.draw(rbind(ggplotGrob(p6),
                               ggplotGrob(p4),
                               ggplotGrob(p2),
                               ggplotGrob(p1),
                               ggplotGrob(p3),
                               ggplotGrob(p5),
                               size = "last") ),
        width = 6,
        height = 6
)

#################
grid.newpage()
plt <- rbind(ggplotGrob(p6),
             ggplotGrob(p4),
             ggplotGrob(p22),
             size = "last")


setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_n25_r14_p25_s5_a.pdf",
        plot = plt,
        width = 4.5,
        height = 4.5
)
# ####################
grid.newpage()
plt <- rbind(ggplotGrob(p1),
             ggplotGrob(p3),
             ggplotGrob(p5),
             size = "last")


setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_n25_r14_p25_s5_b.pdf",
        plot = plt,
        width = 4.5,
        height = 4.5
)
# ####################

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
plt <- grid.draw(rbind(ggplotGrob(p6),
                       ggplotGrob(p2),
                       ggplotGrob(p5),
                       size = "last") )

ggsave( "Intro_reduced_n25_r14_p25_s5.pdf",
        plot = plt,
        width = 4.5, # 5.5
        height = 4.5 # 3.5
)



plt <- ggarrange(p6,p1,p4,p3,p22,p5, ncol=2, nrow=3, common.legend = TRUE, legend="bottom")
setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_reduced_n25_r14_p25_s5_combined.pdf",
        plot = plt,
        width = 8,
        height = 4
)

#######################################
############ lambda sequence ##########
#######################################
# plots
library(tidyverse)
library(ggplot2)
library(latex2exp)
library(grid)

betas <- list("0" = beta_lamba0,
              "0.05" = beta_lamba01,
              "1" = beta_lamba1)

for(j in 1:length(betas)){
  
  # vectorize
  betas[[j]] <- data.frame(
    cbind( as.factor(rep(1:K, each = p)),
           as.vector( betas[[j]][-1,] )
    )
  )
  
  # add names
  betas[[j]] <- cbind( rep(1:p, K), names(betas)[j], betas[[j]] )
  colnames( betas[[j]] ) <- c("Index", "Method", "Task", "Beta")
}

b <- do.call(rbind, betas)
b$Task <- as.factor(b$Task)
# 
# p1 <- ggplot(b, aes(y = Beta, x = Index)) + geom_line() + theme_minimal() + 
#     theme(axis.title.x = element_blank(), axis.text.x = element_blank())


for(j in 1:length(unique(b$Method)) ){
  
  method = unique(b$Method)[j]
  
  #Plot
   if(j == length(unique(b$Method))){
    # make the legend at the bottom here because we use it as the last figure
    assign(paste0("p", j),
           b %>% as_tibble() %>%
             filter(Method == method) %>%
             ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
             # geom_point() +
             geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
             geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
             theme_bw() +
             ylab(TeX(paste0(' $\\delta =$', method) )) + 
             xlab("Coefficient Index") + 
             theme_classic() +
             geom_hline(yintercept = 0, linetype = "dashed") +
             scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5)) + 
             scale_x_continuous(breaks = c(1, 10, 20, 30, 40, 50)) + 
             scale_color_manual(values = c("blue", "red")) +
             theme(plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                   axis.text=element_text(face="bold",color="black", size=rel(1)),
                   axis.title = element_text(face="bold", color="black", size=rel(1)),
                   legend.key.size = unit(2, "line"), # added in to increase size
                   legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                   legend.title = element_text(face="bold", color="black", size = rel(1)),
                   strip.text.x = element_text(face="bold", color="black", size = rel(1)),
                   legend.position="bottom",
                   # legend.justification="right",
                   legend.margin=margin(0,0,0,0),
                   legend.box.margin=margin(-10,-10,-10,-10)
             )
    )
    
    
    
  }else{
    # all other figures
    assign(paste0("p", j),
           b %>% as_tibble() %>%
             filter(Method == method) %>%
             ggplot(aes(x = Index, y = Beta, color= Task)) +  # , group=Price+
             # geom_point() +
             geom_point(alpha = 0.5, data = ~filter(.x, Beta != 0), aes(shape = Task)) +
             geom_segment(alpha = 0.5,  aes(x=Index, xend=Index, y=0, yend=Beta)) +
             theme_bw() +
             ylab(TeX(paste0(' $\\delta =$', method) )) + 
             theme_classic() +
             theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # remove x-axis
             geom_hline(yintercept = 0, linetype = "dashed") +
             scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1.5, 1.5)) + 
             scale_color_manual(values = c("blue", "red")) +
             theme(legend.position="none",
                   plot.title = element_text(hjust = 0.5, color="black", size=rel(1), face="bold"),
                   axis.text=element_text(face="bold",color="black", size=rel(1)),
                   axis.title = element_text(face="bold", color="black", size=rel(1)),
                   legend.key.size = unit(2, "line"), # added in to increase size
                   legend.text = element_text(face="bold", color="black", size = rel(1)), # 3 GCL
                   legend.title = element_text(face="bold", color="black", size = rel(1)),
                   strip.text.x = element_text(face="bold", color="black", size = rel(1))
             )
    )
    
  }
  
}

grid.newpage()
plt <- rbind(ggplotGrob(p1),
                ggplotGrob(p2),
                ggplotGrob(p3),
                size = "last")
                

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_delta_support.pdf",
        plot = plt,
        width = 4.5,
        height = 4.5
)



plt <- ggarrange(p6,p1,p5,p2,p22,p3, ncol=2, nrow=3, common.legend = TRUE, legend="bottom")
setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Intro Support Figure")
ggsave( "Intro_support_combined.pdf",
        plot = plt,
        width = 8,
        height = 4
)

