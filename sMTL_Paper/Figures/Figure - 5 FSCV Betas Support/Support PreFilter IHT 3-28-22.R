library(JuliaConnectoR)
setwd("~/Desktop/Research")

source("sparseFn_iht_test_MT.R") 

full <- read.csv("/Users/gabeloewinger/Desktop/Research Final/Methods Paper/Draft/Figures/Electrode Graphs/sub_samp2500")[,-c(1,2,4,5)]
juliaPath <- "/Applications/Julia-1.5.app/Contents/Resources/julia/bin"
juliaFnPath <- "/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/IHT/Tune/"

Sys.setenv(JULIA_BINDIR = juliaPath)
L0_MS_z3 <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexact_diffAS_tuneTest.jl") ) # sepratae active sets for each study


lambda <- sort( unique( c(1e-6, 1e-5, 0.0001, 0.001, 5,10,
                          exp(-seq(0,5, length = 5))
                        ) ), decreasing = TRUE 
                ) 

source("sparseFn_iht_test.R") # USE TEST VERSION HERE
sparseCV_iht_par <- sparseCV_iht

numCovs <- p <- 1000 # number of features
WSmethod <- 2 
ASpass <- TRUE
rhoStar <- 50 # cardinality

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
                            nfolds = 3,
                            cvFolds = 5,
                            juliaPath = juliaPath,
                            juliaFnPath = juliaFnPath,
                            messageInd = TRUE,
                            LSitr = NA,
                            LSspc = NA,
                            threads = 1,
                            WSmethod = WSmethod,
                            ASpass = ASpass
)

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
                 localIter = 0,
                 WSmethod = WSmethod,
                 ASpass = ASpass
)

write.csv(betas, 
          paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_1000_", rhoStar),
          row.names = FALSE)

betas <- read.csv( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_1000_", rhoStar) )
suppHet(betas, intercept = TRUE)

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
    ylab(TeX('$\\hat{\\mathbf{\\beta}}_k^*$') ) +
    # ylab(TeX('$sgn(\\hat{\\mathbf{\\beta}}_k) * log(|\\hat{\\mathbf{\\beta}}_k|)$') )+ 
    #xlab(TeX('$\\mathbf{Coefficient Index}}$')) + 
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

ggsave( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/carbon_betas_", rhoStar, ".pdf"),
        plot = plt_betas,
        width = 10,
        height = 6
)


###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_1000_", rhoStar) )
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
    ylab(TeX('$\\hat{\\mathbf{\\beta}}_k^*$') ) +
    # ylab(TeX('$sgn(\\hat{\\mathbf{\\beta}}_k) * log(|\\hat{\\mathbf{\\beta}}_k|)$') )+ 
    #xlab(TeX('$\\mathbf{Coefficient Index}}$')) + 
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

ggsave( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/carbon_betas_", rhoStar, "_psub_", p_sub, ".pdf"),
        plot = plt_betas,
        width = 10,
        height = 6
)



##########################
# Zbar method
##########################
lambdaZ <- sort( unique( c(1e-4, 1e-6, 1e-5, 1e-4, 1e-3,
                           exp(-seq(0,5, length = 8)),
                           1:3) ), 
                 decreasing = TRUE )

lambdaBeta <- c( 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 )

tune.grid_MSZ_5 <- as.data.frame(  expand.grid( ridgeLambda / 2, 0, lambdaZ, rhoStar) )

colnames(tune.grid_MSZ_5) <- c("lambda1", "lambda2", "lambda_z","rho")
tune.grid_MSZ_5$lambda1 <- tune.grid_MSZ_5$lambda1 * (tune.grid_MSZ_5$rho / rhoStar)

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
                           hoso = "multiTask", # could balancedCV (study balanced CV necessary if K =2)
                           method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                           nfolds = 3,
                           cvFolds = 5,
                           juliaPath = juliaPath,
                           juliaFnPath = juliaFnPath,
                           messageInd = TRUE,
                           LSitr = NA, 
                           LSspc = NA,
                           threads = 1,
                           WSmethod = WSmethod,
                           ASpass = ASpass
)

MSparams <- tuneMS$best # parameters
lambdaZstar <- MSparams$lambda_z

lambdaZgrid <- c( seq(1.5, 10, length = 5), seq(0.1, 1, length = 5) ) * lambdaZstar
lambdaZgrid <- lambdaZgrid # make sure this is below a threshold to prevent numerical issues
lambdaZgrid <- sort(lambdaZgrid, decreasing = TRUE)

gridUpdate <- as.data.frame(  expand.grid( lambdaBeta, 0, lambdaZgrid, rhoStar) )
colnames(gridUpdate) <- c("lambda1", "lambda2", "lambda_z","rho")

gridUpdate <- gridUpdate[  order(gridUpdate$rho,
                                 gridUpdate$lambda1,
                                 -gridUpdate$lambda_z,
                                 decreasing=TRUE),     ]

gridUpdate <- unique(gridUpdate)

tuneMS <- sparseCV_iht_par(data = full,
                           tune.grid = gridUpdate,
                           hoso = "multiTask", # could balancedCV (study balanced CV necessary if K =2)
                           method = "MS_z", # could be L0 for sparse regression or MS # for multi study
                           nfolds = 3,
                           cvFolds = 5,
                           juliaPath = juliaPath,
                           juliaFnPath = juliaFnPath,
                           messageInd = TRUE,
                           LSitr = NA, 
                           LSspc = NA,
                           threads = 1,
                           WSmethod = WSmethod,
                           ASpass = ASpass
)

# final models

MSparams <- tuneMS$best # parameters

L0_MS_z <- juliaCall("include", paste0(juliaFnPath, "BlockComIHT_inexactAS_tune_old.jl") ) # MT: Need to check it works;  "_tune_old.jl" version gives the original active set version that performs better #\beta - \betaBar penalty

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
                    ASpass = ASpass
)

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
                  ASpass = ASpass
)


write.csv(betas, 
          paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_zBar_1000_", rhoStar),
          row.names = FALSE)

betas <- read.csv(paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_zBar_1000_", rhoStar))
    
suppHet(betas, intercept = TRUE)
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
    ylab(TeX('$\\hat{\\mathbf{\\beta}}_k^*$') ) +
    # ylab(TeX('$sgn(\\hat{\\mathbf{\\beta}}_k) * log(|\\hat{\\mathbf{\\beta}}_k|)$') )+ 
    #xlab(TeX('$\\mathbf{Coefficient Index}}$')) + 
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

ggsave( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/carbon_betas_Zbar_", rhoStar, ".pdf"),
        plot = plt_z_bar,
        width = 10,
        height = 6
)


###########################
# First 100
###########################
p_sub <- 100
betas <- read.csv(paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/betaFits_zBar_1000_", rhoStar))
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
    ylab(TeX('$\\hat{\\mathbf{\\beta}}_k^*$') ) +
    # ylab(TeX('$sgn(\\hat{\\mathbf{\\beta}}_k) * log(|\\hat{\\mathbf{\\beta}}_k|)$') )+ 
    #xlab(TeX('$\\mathbf{Coefficient Index}}$')) + 
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

ggsave( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/fscv/carbon_betas_Zbar_", rhoStar, "_psub_", p_sub, ".pdf"),
        plot = plt_z_bar,
        width = 10,
        height = 6
)


###########################
# Combined First 100
###########################
plt_betas <- plt_betas + ylab(TeX(paste0( "TS-SR  ", "  $\\hat{\\mathbf{\\beta}}_k^*$") ))
plt_z_bar <- plt_z_bar + ylab(TeX(paste0( "Zbar+L2  ", "  $\\hat{\\mathbf{\\beta}}_k^*$") ))

plts_cmb <- ggarrange(plt_betas, plt_z_bar, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")

ggsave( paste0("/Users/gabeloewinger/Desktop/Research Final/Sparse Multi-Study/Figures/Figure - 5 FSCV Betas Support/carbon_betas_", rhoStar, "_psub_", p_sub, "combined.pdf"),
        plot = plts_cmb,
        width = 14, # 12
        height = 5 # 5
)
