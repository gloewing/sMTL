L <- 10 # most important 

studies <- 18
K <- 8
ds <- 1
setwd("~/Desktop/Research")

full <- read.csv("~/Desktop/Research Final/Sparse Multi-Study/GP Data/final/breastCancer_data")


# remove cols not using
XY <- substr(colnames(full), 1, 2) # extract first letter and see which one is Y
Yindx <- which(XY == "Y_") 
Yrm <- Yindx[-seq(1,K)] 
Yindx <- Yindx[1:K]

if( length(Yrm) > 0)   full <- full[,-Yrm]

Xindx <- seq(1, ncol(full) )[-c(1, Yindx)] # all cols except outcome and Study
colnames(full)[Xindx] <- paste0( "x_", colnames(full)[Xindx] ) 
full[,Xindx] <- scale(full[,Xindx],
                      center = TRUE,
                      scale = TRUE
                      )

numCovs <- p <- length(Xindx) # number of features

library(glmnet)

betas <- matrix(nrow = p + 1, ncol = K)
for(j in Yindx){


    # tune and fit task-specific model
    tune.mod <- cv.glmnet(y = as.matrix(full[, j]),
                          x = as.matrix(full[, Xindx]),
                          alpha = 0,
                          intercept = TRUE,
                          family = "gaussian")
    
    
    betas[, j - 1] <- as.vector( coef(tune.mod, exact = TRUE, s = "lambda.min") )
    rm(tune.mod)
}

# save betas
write.csv(betas, 
          "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/bCancer_betas")

# read back in (remove row.numbers)
betas <- read.csv( "~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/bCancer_betas" )[,-1]

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
    gather(key = "Feature") 

# factor levels
betaDensity$Feature <- factor( betaDensity$Feature,
                               levels = paste0("Coefficient ", 1:ncol(betaStar)))

betaDensity2 <-
    betaDensity %>%
    ggplot(aes(x = value, y = Feature, fill = Feature)) +
    geom_density_ridges() +
    theme_ridges() + 
    geom_vline( xintercept = 0, linetype="dotted", size = 1 ) +
    xlab(TeX('Coefficient Estimate Magnitude ($\\hat{\\mathbf{\\beta}}^{(j)}$)') ) +
    ylab( "" ) +
    theme( legend.position = "none",
           panel.grid.major.x = element_blank() ,
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(color="black", size=rel(3))
            ) 

betaDensity2


ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/bCancer_betas_density.pdf"),
        plot = betaDensity2,
        width = 15,
        height = 10
)

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


ggsave( paste0("~/Desktop/Research Final/Sparse Multi-Study/Figures/Density Plots/bCancer_betas_rug.pdf"),
        plot = betaDensity3,
        width = 14,
        height = 8
)

