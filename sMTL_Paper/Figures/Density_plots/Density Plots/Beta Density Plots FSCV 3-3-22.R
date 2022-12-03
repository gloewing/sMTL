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

# write files
write.csv(betas, 
          "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_density_betas")


###############
# plots
###############

# read back in (remove row.numbers)
betas <- read.csv(  "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_density_betas" )[,-1]
betaStar <- betas[-1,] # remove intercept

# like the variance across betas (variance of random effects)
cMat <- cov(t(betas[-1,]))
mean(diag(cMat))

# find most important
indxMax <- sort( rowMeans(abs(betaStar) ),
                 decreasing = TRUE,
                 index.return = TRUE )

indxStar <- indxMax$ix[1:L] # top L most important features

betaStar <- betaStar[indxStar, ] # 

library(ggplot2)
library(latex2exp)
library(ggridges)
library(dplyr)

betaStar <- t(betaStar)
colnames(betaStar) <- paste0("Coefficient ", 1:ncol(betaStar))
betaDensity <-
    betaStar %>% 
    as_tibble() %>%
    tidyr::gather(key = "Feature") 

# factor levels
betaDensity$Feature <- factor( betaDensity$Feature,
                               levels = paste0("Coefficient ", 1:ncol(betaStar)))

# rug plot
betaDensity3 <-
    betaDensity %>%
    ggplot(aes(x = value, y = Feature)) +
    theme_ridges() + 
    geom_vline( xintercept = 0, linetype="dotted", size = 1 ) +
    xlab(TeX('Coefficient Estimate Magnitude ($\\hat{\\mathbf{\\beta}}^{(j)}$)') ) +
    ylab( "" ) +
    theme( legend.position = "none",
           panel.grid.major.x = element_blank() ,
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(color="black", size=rel(2.75))
    ) +
    geom_density_ridges(
        fill = NA,
        color = NA,
        aes(point_color = Feature, point_fill = Feature, point_shape = Feature),
        jittered_points = TRUE,
        position = position_points_jitter(width = 0, height = 0),
        point_shape = '|', point_size = 6, point_alpha = 1, alpha = 0
    ) + 
    scale_point_color_hue(l = 40)

ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_betas_rug.pdf"),
        plot = betaDensity3,
        width = 14,
        height = 8
)

####################################################
# rug plot with 50 most important coefficients
####################################################
L <- 50

# read back in (remove row.numbers)
betas <- read.csv(  "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_density_betas" )[,-1]
betaStar <- betas[-1,] # remove intercept

# like the variance across betas (variance of random effects)
cMat <- cov(t(betas[-1,]))
mean(diag(cMat))

# find most important
indxMax <- sort( rowMeans(abs(betaStar) ),
                 decreasing = TRUE,
                 index.return = TRUE )

indxStar <- indxMax$ix[1:L] # top L most important features

betaStar <- betaStar[indxStar, ] # 


betaStar <- t(betaStar)
colnames(betaStar) <- paste0("Coefficient ", 1:ncol(betaStar))
betaDensity <-
    betaStar %>% 
    as_tibble() %>%
    tidyr::gather(key = "Feature") 

# factor levels
betaDensity$Feature <- factor( betaDensity$Feature,
                               levels = paste0("Coefficient ", 1:ncol(betaStar)))


# rug plot
betaDensity4 <-
    betaDensity %>%
    ggplot(aes(x = value, y = Feature)) +
    theme_ridges() + 
    geom_vline( xintercept = 0, linetype="dotted", size = 1 ) +
    xlab(TeX('Coefficient Estimate Magnitude ($\\hat{\\mathbf{\\beta}}^{(j)}$)') ) +
    ylab( "" ) +
    theme( legend.position = "none",
           panel.grid.major.x = element_blank() ,
           axis.text.y =element_text(face="bold",color="black", size=rel(1.5)),
           axis.text.x =element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(color="black", size=rel(2.75))
    ) +
    geom_density_ridges(
        fill = NA,
        color = NA,
        aes(point_color = Feature, point_fill = Feature, point_shape = Feature),
        jittered_points = TRUE,
        position = position_points_jitter(width = 0, height = 0),
        point_shape = '|', point_size = 6, point_alpha = 1, alpha = 0
    ) + 
    scale_point_color_hue(l = 40)

betaDensity4

ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_betas_rug_50coefs.pdf"),
        plot = betaDensity4,
        width = 14,
        height = 20
)



####################################################
# rug plot with 100 RANDOM coefficients
####################################################
set.seed(1)
L <- 100

# read back in (remove row.numbers)
betas <- read.csv(  "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_density_betas" )[,-1]
betaStar <- betas[-1,] # remove intercept

# like the variance across betas (variance of random effects)
cMat <- cov(t(betas[-1,]))

indxStar <- sample.int(nrow(betaStar), L, replace = FALSE) # random sample of indices

betaStar <- betaStar[indxStar, ] # 


betaStar <- t(betaStar)
colnames(betaStar) <- paste0("Coefficient ", 1:ncol(betaStar))
betaDensity <-
    betaStar %>% 
    as_tibble() %>%
    tidyr::gather(key = "Feature") 

# factor levels
betaDensity$Feature <- factor( betaDensity$Feature,
                               levels = paste0("Coefficient ", 1:ncol(betaStar)))


# rug plot
betaDensity5 <-
    betaDensity %>%
    ggplot(aes(x = value, y = Feature)) +
    theme_ridges() + 
    geom_vline( xintercept = 0, linetype="dotted", size = 1 ) +
    xlab(TeX('Coefficient Estimate Magnitude ($\\hat{\\mathbf{\\beta}}^{(j)}$)') ) +
    ylab( "" ) +
    theme( legend.position = "none",
           panel.grid.major.x = element_blank() ,
           axis.text.y =element_text(face="bold",color="black", size=rel(1)),
           axis.text.x =element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(color="black", size=rel(2.75))
    ) +
    geom_density_ridges(
        fill = NA,
        color = NA,
        aes(point_color = Feature, point_fill = Feature, point_shape = Feature),
        jittered_points = TRUE,
        position = position_points_jitter(width = 0, height = 0),
        point_shape = '|', point_size = 6, point_alpha = 1, alpha = 0
    ) + 
    scale_point_color_hue(l = 40)

betaDensity5

ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/fscv_betas_rug_randomCoefs.pdf"),
        plot = betaDensity5,
        width = 14,
        height = 20
)

