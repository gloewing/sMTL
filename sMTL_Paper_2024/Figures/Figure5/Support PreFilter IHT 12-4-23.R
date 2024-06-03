library(JuliaConnectoR)
library(ggplot2)
library(latex2exp)
setwd("~/Desktop/Research")

full <- read.csv("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/fscv/sub_samp2500")
full <- subset(full, select=-c(X.2, X.1, X, Channel, pH)) # remove unwanted variables

# select random subset
set.seed(1)

juliaPath <- "/Applications/Julia-1.7.app/Contents/Resources/julia/bin"
juliaFnPath_MT <- juliaFnPath <- "/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/IHT/Tune/"
wd <- "/Users/loewingergc/Downloads/sMTL-main/sMTL_Paper/sMTL_Functions/"#"/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune/"


Sys.setenv(JULIA_BINDIR = juliaPath)
L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study


lambda <- sort( unique( c(exp(seq(0, 6, length = 50)),
                          exp(-seq(0, 25, length = 50)) ) ), decreasing = TRUE ) 

lambdaZ <- sort( unique( c(0, 
                           exp(seq(0,4.75, length = 50)),
                           exp(-seq(0,25, length = 50))) ),
                 decreasing = TRUE ) 

source(paste0(wd, "sparseFn_iht_test_MT.R")) # USE TEST VERSION HERE
source(paste0(wd, "sparseFn_iht_test.R")) # USE TEST VERSION HERE

sparseCV_iht_par <- sparseCV_iht

numCovs <- p <- 1000 # number of features
WSmethod <- 2 
ASpass <- TRUE
rhoStar <- rho <- 25 # cardinality
nfolds <- 3

colnames(full)[1] <- "Study"
colnames(full)[2] <- "Y"

# just choose the first 4 arbitrarily for demonstration
full <- full[full$Study %in% seq(1,4), ]
K <- length(unique(full$Study))
Xindx <- seq(1, ncol(full))[-c(1,2)]
Yindx <- which(colnames(full) == "Y")
# 
for(kk in 1:K){

    # indices
    kkIndx <- which(full$Study == kk) # train set indices
    full[kkIndx, Xindx] <- scale(full[kkIndx, Xindx])

}

b <- matrix(0, ncol = K, nrow = numCovs + 1)

###############################
# Separate L0 regressions Plot
###############################
tune.grid_OSE <- expand.grid(lambda1 = unique(lambda),
                             lambda2 = 0,
                             lambda_z = 0,
                             rho = rhoStar)
# 
tune.grid_OSE <- unique( tune.grid_OSE )

L0_tune <- sparseCV_iht_par(data = full,
                            tune.grid = tune.grid_OSE,
                            hoso = "multiTask", # could balancedCV (study balanced CV necessary if K =2)
                            method = "MS_z3", #"MS_z_fast", # this does not borrow information across the active sets
                            nfolds = nfolds,
                            juliaPath = juliaPath,
                            juliaFnPath = juliaFnPath,
                            messageInd = TRUE,
                            LSitr = NA,
                            LSspc = NA,
                            threads = 1,
                            WSmethod = WSmethod,
                            ASpass = ASpass)

L0_tune <- L0_tune$best # parameters
ridgeLambda <- L0_tune$lambda1

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
                 localIter = 0,
                 WSmethod = WSmethod,
                 ASpass = ASpass)

# if running the first time
# write.csv(betas,
#           paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_", p ,"_", rhoStar),
#           row.names = FALSE)

betas <- read.csv( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_1000_", rhoStar) )
suppHet(betas, intercept = TRUE)
l0l2_supp <- suppHet(betas, intercept = TRUE)[1]

betas <- as.matrix( betas[-1,] ) # remove intercept since irrelevant for comparison
library(ggplot2)
library(latex2exp)

p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_betas = 
    ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
    geom_line() + 
    ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
    xlab(('Coefficient Index')) + 
    scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    scale_x_continuous(breaks = c(1, seq(200,1000,200))) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), 
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_", rhoStar, ".pdf"),
        plot = plt_betas,
        width = 10,
        height = 6)

plt_betas_full <- plt_betas

###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_1000_", rhoStar) )
betas <- as.matrix(betas[2:(p_sub +1),]) # first p_sub of coefficients
p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_betas = 
    ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
    geom_line() + 
    ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
    xlab(('Coefficient Index')) + 
    scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    scale_x_continuous(breaks = c(1, seq(50,200,50))) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), 
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_", rhoStar, "_psub_", p_sub, ".pdf"),
        plot = plt_betas,
        width = 10,
        height = 6)

##########################
# Zbar method
##########################
lambdaZ <- lambda

tune.grid_MSZ_5 <- as.data.frame(  expand.grid( ridgeLambda / 2, 0, lambdaZ, rhoStar) )

colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
tune.grid_MSZ_5$lambda1 <- tune.grid_MSZ_5$lambda1 * (tune.grid_MSZ_5$rho / rhoStar)

# order correctly
tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                           -tune.grid_MSZ_5$lambda_z,
                                           decreasing=TRUE),     ]

tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)

# tune z - zbar and rho
# tune multi-study with l0 penalty with z - zbar and beta - betaBar penalties
tuneMS <- sparseCV_iht_par(data = full,
                           tune.grid = tune.grid_MSZ_5,
                           hoso = "multiTask", # could balancedCV (study balanced CV necessary if K =2)
                           method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                           nfolds = nfolds,
                           juliaPath = juliaPath,
                           juliaFnPath = juliaFnPath,
                           messageInd = TRUE,
                           LSitr = 10, 
                           LSspc = 1,
                           threads = 1,
                           WSmethod = WSmethod,
                           ASpass = ASpass)

MSparams <- tuneMS$best # parameters
lambdaZstar <- MSparams$lambda_z

MSparams <- tuneMS$best # parameters

L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty

b_init <- matrix(0, ncol = K, nrow = numCovs + 1)

# warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                    y = as.vector(full$Y),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                    study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                    beta = b_init,
                    lambda1 = MSparams$lambda1,
                    lambda2 = MSparams$lambda2,
                    lambda_z = 0,
                    scale = TRUE,
                    maxIter = 10000,
                    localIter = 0,
                    WSmethod = WSmethod,
                    ASpass = ASpass)

# final model
betas = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
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

# 
# write.csv(betas,
#           paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_zBar_1000_", rhoStar),
#           row.names = FALSE)

betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_zBar_1000_", rhoStar))
    
suppHet(betas, intercept = TRUE)
zbar_supp <- suppHet(betas, intercept = TRUE)[1]
zbar_supp / l0l2_supp

betas <- as.matrix(betas[-1,]) # remove intercept since irrelevant for comparison
library(ggplot2)
library(latex2exp)

p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_z_bar = 
    ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
    geom_line() + 
    ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
    xlab(('Coefficient Index')) + 
    scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    scale_x_continuous(breaks = c(1, seq(200,1000,200))) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), 
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_Zbar_", rhoStar, ".pdf"),
        plot = plt_z_bar,
        width = 10,
        height = 6)

plt_z_bar_full <- plt_z_bar

###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_zBar_1000_", rhoStar))
betas <- as.matrix(betas[2:(p_sub +1),]) # first p_sub of coefficients
p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_z_bar = 
    ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
    geom_line() + 
    ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
    xlab(('Coefficient Index')) + 
    scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    scale_x_continuous(breaks = c(1, seq(50,200,50))) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), 
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_Zbar_", rhoStar, "_psub_", p_sub, ".pdf"),
        plot = plt_z_bar,
        width = 10,
        height = 6)

# -------------------------------------------------------

##########################
# Bbar method
##########################
lambdaB <- lambda

tune.grid_MSZ_5 <- as.data.frame(  expand.grid( 0, lambdaB, 0, rhoStar) )

colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
tune.grid_MSZ_5$lambda1 <- tune.grid_MSZ_5$lambda1 * (tune.grid_MSZ_5$rho / rhoStar)

# order correctly
tune.grid_MSZ_5 <- tune.grid_MSZ_5[  order(-tune.grid_MSZ_5$rho,
                                           -tune.grid_MSZ_5$lambda2,
                                           decreasing=TRUE),     ]

tune.grid_MSZ_5 <- unique(tune.grid_MSZ_5)

# tune z - zbar and rho
# tune multi-study with l0 penalty with z - zbar and beta - betaBar penalties
tuneMS <- sparseCV_iht_par(data = full,
                           tune.grid = tune.grid_MSZ_5,
                           hoso = "multiTask", # could balancedCV (study balanced CV necessary if K =2)
                           method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                           nfolds = nfolds,
                           juliaPath = juliaPath,
                           juliaFnPath = juliaFnPath,
                           messageInd = TRUE,
                           LSitr = 10, 
                           LSspc = 1,
                           threads = 1,
                           WSmethod = WSmethod,
                           ASpass = ASpass)

MSparams <- tuneMS$best # parameters
lambdaZstar <- MSparams$lambda_z

MSparams <- tuneMS$best # parameters

L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty

b_init <- matrix(0, ncol = K, nrow = numCovs + 1)

# warm start with OSE L0 (i.e., lambda_z = 0 and tuned lambda1/lambda2)
warmStart = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
                    y = as.vector(full$Y),
                    rho = min( c(MSparams$rho * 4, numCovs - 1) ),
                    study = as.vector( full$Study ), # these are the study labels ordered appropriately for this fold
                    beta = b_init,
                    lambda1 = MSparams$lambda1,
                    lambda2 = MSparams$lambda2,
                    lambda_z = 0,
                    scale = TRUE,
                    maxIter = 10000,
                    localIter = 0,
                    WSmethod = WSmethod,
                    ASpass = ASpass)

# final model
betas = L0_MS_z(X = as.matrix( full[ , -c(1,2) ]) ,
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

# 
# write.csv(betas,
#           paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_BBar_1000_", rhoStar),
#           row.names = FALSE)

betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_BBar_1000_", rhoStar))

suppHet(betas, intercept = TRUE)
bbar_supp <- suppHet(betas, intercept = TRUE)[1]
bbar_supp / l0l2_supp

betas <- as.matrix(betas[-1,]) # remove intercept since irrelevant for comparison
library(ggplot2)
library(latex2exp)

p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_b_bar = 
  ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
  geom_line() + 
  ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
  xlab(('Coefficient Index')) + 
  scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
  scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
  theme_classic(base_size = 12) +
  scale_x_continuous(breaks = c(1, seq(200,1000,200))) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(2)),
         axis.title = element_text(face="bold", color="black", size=rel(2)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(2)), 
         legend.title = element_text(face="bold", color="black", size = rel(2)),
         strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
  ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_Bbar_", rhoStar, ".pdf"),
        plot = plt_b_bar,
        width = 10,
        height = 6)

plt_b_bar_full <- plt_b_bar

###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_BBar_1000_", rhoStar))
betas <- as.matrix(betas[2:(p_sub +1),]) # first p_sub of coefficients
p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_b_bar = 
  ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
  geom_line() + 
  ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
  xlab(('Coefficient Index')) + 
  scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
  scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
  theme_classic(base_size = 12) +
  scale_x_continuous(breaks = c(1, seq(50,200,50))) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(2)),
         axis.title = element_text(face="bold", color="black", size=rel(2)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(2)), 
         legend.title = element_text(face="bold", color="black", size = rel(2)),
         strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
  ) + guides(fill= guide_legend(title="Study"))  

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_Bbar_", rhoStar, "_psub_", p_sub, ".pdf"),
        plot = plt_b_bar,
        width = 10,
        height = 6)
# ###########################
# # Combined First 100
# ###########################
# plt_betas <- plt_betas + ylab(TeX(paste0( "$L_0 L_2$  ", "  $\\hat{{\\beta}}_k^*$") ))
# plt_z_bar <- plt_z_bar + ylab(TeX(paste0( "Zbar+L2  ", "  $\\hat{{\\beta}}_k^*$") ))
# 
# plts_cmb <- ggarrange(plt_betas, plt_z_bar, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")
# 
# ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_", rhoStar, "_psub_", p_sub, "combined.pdf"),
#         plot = plts_cmb,
#         width = 14, # 12
#         height = 5)



#######################
# gel
#######################
library(grpreg)

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

# convert into form used by `sparsegl` package
Xlist <- lapply(Xlist, function(x) as.matrix(x[,-1]) ) # make matrix and remove column of 1s for intercept
Xlist <- Matrix::bdiag(Xlist)
Ymeans <- unlist(lapply(Ylist, mean)) #colMeans(full[,Yindx]) # means for intercept
Ylist <- lapply(Ylist, function(x) scale(x, center = TRUE, scale = FALSE)) # center Ys for each task
Ylist <- as.numeric(do.call(c, Ylist))
groupIDs <- rep(1:numCovs, K)
tune_length <- 100

# hyperparameter tuning
Xlist <- as.matrix(Xlist)
set.seed(1)

asprse_mat <- data.frame(matrix(NA, nrow = 1, ncol = 2)) 
colnames(asprse_mat) <- c("rmse", "lambda")
beta_vec <- vector(length = ncol(Xlist) + 1) # store best models

# tune model
# hyperparameter tuning
tuneMS <- cv.grpreg(X = Xlist,
                    y = Ylist,
                    penalty =  "gel", 
                    family = "gaussian",
                    group = groupIDs,
                    intercept = FALSE,
                    nfolds = nfolds,
                    lambda.min = 1e-12, # needed to achieve solution sparsity in this example
                    eps=1e-2, # needed to achieve solution sparsity in this example
                    nlambda = tune_length*500, # ensure enough values at the specific rho value
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
betas <- rbind(Ymeans, beta_sprgl) # add intercepts back on



## plotting
# write.csv(betas,
#           paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_gel_1000_", rhoStar),
#           row.names = FALSE)

betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_gel_1000_", rhoStar))

gel_supp <- suppHet(betas, intercept = TRUE)[1]
gel_supp / l0l2_supp
betas <- as.matrix(betas[-1,]) # remove intercept since irrelevant for comparison

p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_gel = 
  ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
  geom_line() + 
  ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
  xlab(('Coefficient Index')) + 
  scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
  scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
  theme_classic(base_size = 12) +
  scale_x_continuous(breaks = c(1, seq(200,1000,200))) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(2)),
         axis.title = element_text(face="bold", color="black", size=rel(2)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(2)), 
         legend.title = element_text(face="bold", color="black", size = rel(2)),
         strip.text.x = element_text(face="bold", color="black", size = rel(2.5)) ) + 
  guides(fill= guide_legend(title="Study"))  

plt_gel_full <- plt_gel

###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv(paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/betaFits_gel_1000_", rhoStar))
betas <- as.matrix(betas[2:(p_sub +1),]) # first p_sub of coefficients
p <- nrow(betas)
betaIndx <- seq(1, p )
betaIndx <- rep(betaIndx, K)
elec <- rep( seq(1,K),  each = p)

# rescale
b = sign(betas) * log( abs(betas) + 1e-20 ) * (betas != 0)

mat <- as.data.frame( cbind( betaIndx, as.vector(betas), as.factor(elec), as.vector(b) ) )
colnames(mat) <- c("Feature", "Betas", "Task", "logBetas")
mat$Task <- as.factor(mat$Task)

plt_gel = 
  ggplot(mat, aes( y = logBetas, x = Feature, color = Task )) +
  geom_line() + 
  ylab(TeX('$\\hat{{\\beta}}_k^*$') ) +
  xlab(('Coefficient Index')) + 
  scale_fill_manual(values = c("#ca0020", "black", "#E69F00", "#525252") ) +
  scale_color_manual(values = c("#ca0020", "black", "#0868ac", "#E69F00", "#525252")) +
  theme_classic(base_size = 12) +
  scale_x_continuous(breaks = c(1, seq(50,200,50))) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(2)),
         axis.title = element_text(face="bold", color="black", size=rel(2)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(2)), 
         legend.title = element_text(face="bold", color="black", size = rel(2)),
         strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
  ) + guides(fill= guide_legend(title="Study"))  



###########################
# Combined First 100
###########################
plt_betas <- plt_betas + ylab(TeX(paste0( "$L_0 L_2$  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_z_bar <- plt_z_bar + ylab(TeX(paste0( "Zbar+L2  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_b_bar <- plt_b_bar + ylab(TeX(paste0( "$Bbar$  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_gel <- plt_gel + ylab(TeX(paste0( "gel  ", "  $\\hat{{\\beta}}_k^*$") ))


plts_cmb <- ggpubr::ggarrange(plt_betas, plt_z_bar, plt_gel, plt_b_bar, ncol=2, nrow=2, common.legend = TRUE, legend="bottom")

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_", rhoStar, "_psub_", p_sub, "combined.pdf"),
        plot = plts_cmb,
        width = 14, # 12
        height = 10)


###########################
# Combined Full
###########################
plt_betas <- plt_betas_full + ylab(TeX(paste0( "$L_0 L_2$  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_z_bar <- plt_z_bar_full + ylab(TeX(paste0( "Zbar+L2  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_bbar <- plt_b_bar_full + ylab(TeX(paste0( "$Bbar$  ", "  $\\hat{{\\beta}}_k^*$") ))
plt_gel <- plt_gel_full + ylab(TeX(paste0( "gel  ", "  $\\hat{{\\beta}}_k^*$") ))


plts_cmb <- ggpubr::ggarrange(plt_betas, plt_z_bar, plt_gel, plt_bbar, ncol=2, nrow=2, common.legend = TRUE, legend="bottom")

ggsave( paste0("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/Resubmission/carbon_betas_", rhoStar, "_combined.pdf"),
        plot = plts_cmb,
        width = 14, # 12
        height = 10)
