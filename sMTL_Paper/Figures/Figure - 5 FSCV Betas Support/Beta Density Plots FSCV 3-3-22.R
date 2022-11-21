L <- 10 # most important 
numCovs <- p <- 1000 # number of features

full <- read.csv("/Users/gabeloewinger/Desktop/Research Final/Methods Paper/Draft/Figures/Electrode Graphs/sub_samp2500")[,-c(1,2,4,5)]

colnames(full)[1] <- "Study"
colnames(full)[2] <- "Y"

# just choose the first 4 arbitrarily for demonstration
#full <- full[full$Study %in% seq(1,4), ]
K <- length(unique(full$Study))
Xindx <- seq(1, ncol(full))[-c(1,2)]
Yindx <- which(colnames(full) == "Y")
library(glmnet)

betas <- matrix(nrow = p + 1, ncol = K)
for(j in 1:K){
    
    # rows for this task
    indx <- which(full$Study == j)
    
    # scale design matrix
    X <- scale(as.matrix(full[indx, Xindx]),
               center = TRUE,
               scale = TRUE
               )
    
    # tune and fit task-specific model
    tune.mod <- cv.glmnet(y = as.matrix(full[indx, Yindx]),
                          x = X,
                          alpha = 0,
                          intercept = TRUE,
                          family = "gaussian")
    
    
    betas[,j] <- as.vector( coef(tune.mod, exact = TRUE, s = "lambda.min") )
    rm(X, tune.mod)
}

set.seed(5)
betaStar <- betas[-1,] # remove intercept

# find most important
# indxMax <- sort( rowMeans(abs(betaStar) ), 
#                  decreasing = TRUE, 
#                  index.return = TRUE )
# 
# indxStar <- indxMax$ix[1:L] # top L most important features

indxStar <- sample(1:1000, 10)
betaStar <- betaStar[indxStar, ] # 



# plot(density( betaStar[10,] ))
# plot(density( betas[1000,] ))
library(ggplot2)
library(latex2exp)
library(ggridges)
library(dplyr)

betaStar <- t(betaStar)
colnames(betaStar) <- paste0("Feature ", 1:ncol(betaStar))
betaDensity <-
    betaStar %>% 
    as_tibble() %>%
    gather(key = "Feature") 

# factor levels
betaDensity$Feature <- factor( betaDensity$Feature,
                               levels = paste0("Feature ", 1:ncol(betaStar)))

betaDensity <-
    betaDensity %>%
        ggplot(aes(x = value, y = Feature, fill = Feature)) +
            geom_density_ridges() +
            theme_ridges() + 
            theme(legend.position = "none")

betaDensity

ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_betas_density.pdf"),
        plot = betaDensity,
        width = 15,
        height = 10
)
