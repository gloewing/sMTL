# support figures
#######################
# compare different K 
#######################
setwd("~/Desktop/Research/cf_MT7")

library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)

Kvec <- 4
pVec <- c(1, 5, 10, 25)
ls_prob <- ls_pair <- ls_beta <- vector(length = length(Kvec), "list")
rho <- rhoVec <- c(5, 10, 25, 50, 100, 200, 500)
Kvec <- 4
n <- 1000
s <- 500 # rho max: max features
itrs=100
cnt <- 0
rmseMat <- matrix(nc = 4, nr = itrs)

for(p in pVec){
    
    for(k in Kvec){
        flNm <-  paste0("cfFSCV_MT_supp__datPro_2_sclX_TRUE_rhoLen_7_totSims_100_L0sseTn_sse_rdgLmd_TRUE_MSTn_multiTask_Yscl_TRUE_rhoMax_", s, "_nFld_5_LSitr_NA_LSspc_NA_fitLocal_0_Zlen_14_wsMeth_2_asPass_TRUE_TnIn_TRUE_colSub_", p, "_n_k_", n, "_K_", k)
        if(file.exists(flNm)){
            # check to see if file exists
            cnt <- cnt + 1
            d <- read.csv(flNm) 
            
            # remove NAs
            ind <- apply(d, 1, function(x) all(is.na(x)))
            d <- d[!ind,]
            if(nrow(d) > 2){
                for(rho in 1:length(rhoVec)){
                    cnt <- cnt + 1
                    
                    rmseMat <- data.frame("LASSO" = d[paste0("MT_mtLasso_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "CS+Bbar" = d[paste0("MT_msP1_L0_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "Zbar+L2" = d[paste0("MT_msP2_L0_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "CS+L2" = d[paste0("MT_msP3_L0_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "Bbar" = d[paste0("MT_msP4_L0_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "Zbar+Bbar" = d[paste0("MT_msP5_L0_", rho)] / d[paste0( "MT_oseL0_", rho)],
                                          "CVX_Bbar" = d[paste0("msP1_con")] / d[paste0( "MT_oseL0_", rho)],
                                          "Ridge" = d[paste0("msP3_con")] / d[paste0( "MT_oseL0_", rho)],
                                          "Trace" = d[paste0("MTL_trace")] /d[paste0( "MT_oseL0_", rho)],
                                          "LASSO_full"= d[paste0( "MT_lasso_low")] / d[paste0( "MT_oseL0_", rho)] # tuned despite name
                    )
                    
                    #names(rmseMat) <- gsub(x = names(rmseMat), pattern = paste0("\\_", rho), replacement = "")
                    names(rmseMat) <- c("LASSO", "CS+Bbar", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar", "CVX_Bbar", "Ridge", "Trace", "LASSO_full")
                   
                    
                    pairMat <- data.frame("LASSO" = d[paste0("MT_mtLasso_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")],
                                          "CS+Bbar" = d[paste0("MT_msP1_L0_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")],
                                          "Zbar+L2" = d[paste0("MT_msP2_L0_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")],
                                          "CS+L2" = d[paste0("MT_msP3_L0_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")],
                                          "Bbar" = d[paste0("MT_msP4_L0_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")],
                                          "Zbar+Bbar" = d[paste0("MT_msP5_L0_", rho, "_pair")] / d[paste0( "MT_oseL0_", rho, "_pair")]
                    )
                    
                    names(pairMat) <- c("LASSO", "CS+Bbar", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar")
                    
                    
                    probMat <- data.frame("LASSO" = d[paste0("MT_mtLasso_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")],
                                          "CS+Bbar" = d[paste0("MT_msP1_L0_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")],
                                          "Zbar+L2" = d[paste0("MT_msP2_L0_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")],
                                          "CS+L2" = d[paste0("MT_msP3_L0_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")],
                                          "Bbar" = d[paste0("MT_msP4_L0_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")],
                                          "Zbar+Bbar" = d[paste0("MT_msP5_L0_", rho, "_prob")] / d[paste0( "MT_oseL0_", rho, "_prob")]
                    )
                    
                    names(probMat) <- c("LASSO", "CS+Bbar", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar")
                    
                    r <- rhoVec[rho]
                    
                    ls_beta[[cnt]] <- cbind( 
                        gather(rmseMat), 
                        k, p, r
                    )
                    
                    ls_pair[[cnt]] <- cbind( 
                        gather(pairMat), 
                        k, p, r
                    )
                    
                    ls_prob[[cnt]] <- cbind( 
                        gather(probMat), 
                        k, p, r
                    )
                    
                    # ls_supp[[cnt]] <- cbind( 
                    #     gather(suppMat), 
                    #     k, p, rhoVec[rho]
                    # )
                }
                d1 <- d
                
                rm(d)
            }
        }
        
    }
}


dat <- do.call(rbind, ls_beta)
dat$k <- as.factor(dat$k)
dat$r <- as.factor(dat$r)
# dat$p <- as.factor(dat$p)
dat$p <- as.factor(1000 / dat$p)
dat$key <- as.factor(dat$key)

#########################
# rmse -- only L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
        key %in% c("LASSO", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar", "Ridge", "LASSO_full"), # , "LASSO", "Trace"
        
                p == 1000 
                  ) %>%
    ggplot(aes( y = value, x = r, fill = key )) +
   # facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{TS-SR}$') )+ 
    xlab("Number of Features (p)") + 
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.25, 1.0) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  #guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_L0_support.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)


dat <- do.call(rbind, ls_pair)
dat$k <- as.factor(dat$k)
dat$p <- as.factor(dat$p)
dat$r <- as.factor(dat$r)
dat$key <- as.factor(dat$key)
#########################
# pair -- non-L0 methods
#########################
# common support vs. mrg and ose
plt_pair = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
                  key %in% c("LASSO", "CS+Bbar", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar"), # , 
    ) %>%
    ggplot(aes( y = value, x = p, fill = key )) +
    facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{TS-SR}$') )+ 
    xlab("s") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0, 20) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  #guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_all_pair.pdf",
        plot = plt_pair,
        width = 15,
        height = 7
)


dat <- do.call(rbind, ls_prob)
dat$k <- as.factor(dat$k)
dat$p <- as.factor(dat$p)
dat$r <- as.factor(dat$r)
dat$key <- as.factor(dat$key)
#########################
# rmse -- non-L0 methods
#########################
# common support vs. mrg and ose
plt_prob = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
        key %in% c("LASSO", "CS+Bbar", "Zbar+L2", "CS+L2", "Bbar", "Zbar+Bbar"), # , 
    ) %>%
    ggplot(aes( y = value, x = p, fill = key )) +
    facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    #ylim(0, 2) +
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{TS-SR}$') )+ 
    xlab("s") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0, 1000) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  #guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_all_prob.pdf",
        plot = plt_pair,
        width = 15,
        height = 7
)
