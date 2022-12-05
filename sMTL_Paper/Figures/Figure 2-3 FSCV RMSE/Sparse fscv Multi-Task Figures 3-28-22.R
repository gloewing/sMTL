# Figures and Tables for FSCV
f1score <- function(data, name){
    tp <- data[ paste0(name, "_tp") ]
    fp <- data[ paste0(name, "_fp") ]
    
    return( tp / (tp + 0.5 * (fp + 1 - tp ) ) ) # f1 score
}

######################################################
# compare different K and betaVar -- no regularization
######################################################
setwd("~/Desktop/Research/cf_MT7")
library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)

Kvec <- c(4)
nVec <- c(250,500,1000,5000)
rVec <- c(1, 5, 10, 25) # colsubSamp
sVec <- c(25) # rho max: max features
ls_f1 <- ls_supp <- ls_tp <- ls_fp <- ls_beta <- ls_coef <- vector(length = length(nVec) * length(rVec), "list")

itrs <- 100 # number of simulation iterations
cnt <- 0
rmseMat <- matrix(nc = 4, nr = itrs)
for(r in rVec){
    for(s in sVec){
        for(n in nVec){
        for(k in Kvec){
            flNm <-  paste0("cfFSCV_MT_datPro_2_sclX_TRUE_rhoLen_5_totSims_100_L0sseTn_sse_rdgLmd_TRUE_MSTn_multiTask_Yscl_TRUE_rhoMax_", s, "_nFld_5_LSitr_NA_LSspc_NA_fitLocal_0_Zlen_14_wsMeth_2_asPass_TRUE_TnIn_TRUE_colSub_", r, "_n_k_", n, "_K_", k)
                            #""              # 
                    if(file.exists(flNm)){
                        # check to see if file exists
                        cnt <- cnt + 1
                        d <- read.csv(flNm) 
                        
                        # remove NAs
                        ind <- apply(d, 1, function(x) all(is.na(x)))
                        d <- d[!ind,]
                        
                        rmseMat <- data.frame(#"LASSO" = d$MTL_L2L1 / d$MT_oseL0,
                                              #"Ridge" = d$MT_mrgRdg / d$MT_oseL0,
                                              "mrgL0" = d$mrgL0  / d$MT_oseL0,
                                              "CS+Bbar" = d$MT_msP1_L0 / d$MT_oseL0,
                                              "Zbar+L2" = d$MT_msP2_L0 / d$MT_oseL0,
                                              "CS+L2" = d$MT_msP3_L0 / d$MT_oseL0,
                                              "Bbar" = d$MT_msP4_L0 / d$MT_oseL0,
                                              "Zbar+Bbar" = d$MT_msP5_L0 / d$MT_oseL0,
                                              "CVX_Bbar" = d$MT_msP1_con / d$MT_oseL0,
                                              "Ridge" = d$MT_msP3_con / d$MT_oseL0,
                                              "Trace" = d$MTL_trace / d$MT_oseL0,
                                              "LASSO"= d$MT_lasso_low_s / d$MT_oseL0
                        )
                        
                        names(rmseMat) <- gsub(x = names(rmseMat), pattern = "\\.", replacement = "+")

                        suppMat <- data.frame(
                                              "CS+Bbar" = d$s_MS1 - d$s_ose,
                                              "Zbar+L2" = d$s_MS2 - d$s_ose,
                                              "CS+L2" = d$s_MS3 - d$s_ose,
                                              "Bbar" = d$s_MS4 - d$s_ose,
                                              "Zbar+Bbar" = d$s_MS5 - d$s_ose,
                                              #"LASSO" = d$MTL_L2L1_supp - d$s_ose,
                                              "LASSO"= d$MT_lasso_low - d$s_ose
                                              
                        )
                        
                        names(suppMat) <- gsub(x = names(suppMat), pattern = "\\.", replacement = "+")
                        
                        
                        ls_beta[[cnt]] <- cbind( 
                            gather(rmseMat), 
                            k,n,r,s
                        )
                        
                        ls_supp[[cnt]] <- cbind( 
                            gather(suppMat), 
                            k,n, r,s
                        )
                        
                        d1 <- d
                        
                        rm(d)
                    }
                    
                    
                    
                }
            }
}
}


dat <- do.call(rbind, ls_beta)
dat$r <- as.factor(1000 / dat$r)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r <- as.factor(dat$r)
dat$n <- as.factor(dat$n)

#########################
# rmse -- only L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(n %in% c(250, 1000),
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2") ) %>%
    ggplot(aes( y = value, x = r, fill = key )) +
    facet_wrap( ~ n, nrow = 1) +
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
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.875, 1.25) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_L0.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)

plt_rmse = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(n %in% c(250),
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2") ) %>%
    ggplot(aes( y = value, x = r, fill = key )) +
    # facet_wrap( ~ n, nrow = 1) +
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
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.875, 1.25) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_L0_n1000.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)

#########################
# rmse -- non-L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  # 
    dplyr::filter(n %in% c(250, 1000),
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2")
    ) %>%
    ggplot(aes( y = value, x = r, fill = key )) +
    facet_wrap( ~ n, nrow = 1) +
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
    xlab(TeX('$\\mathbf{p}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.6, 1.25) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskRMSE_fscv_all.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)

###########################################

###########################################
dat <- do.call(rbind, ls_supp)
dat$r <- as.factor(1000 / dat$r)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r <- as.factor(dat$r)
dat$n <- as.factor(dat$n)

#########################
# supp
#########################
# common support vs. mrg and ose
plt_supp = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(! n %in% c(250, 1000),
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2")
    ) %>%
    dplyr::group_by(key, r, n) %>% 
    dplyr::summarize(my_mean = mean(value, na.rm = TRUE) ) %>% 
    arrange(r, n) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = r, fill = key)) +
    facet_wrap( ~ n, nrow = 1) +
    geom_bar(stat="identity", position=position_dodge()) +
    ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{p}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    #ylim(0.8, 1.1) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))
                                     
setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskSuppSize_fscv.pdf",
        plot = plt_supp,
        width = 15,
        height = 7
)

#########################
# supp
#########################
# common support vs. mrg and ose
plt_supp = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(! n %in% c(2000, 5000),
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2")
    ) %>%
    dplyr::group_by(key, r, n) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(r, n) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = r, fill = key)) +
    facet_wrap( ~ n, nrow = 1) +
    geom_bar(stat="identity", position=position_dodge()) +
    ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{p}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    #ylim(0.8, 1.1) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))#
                                     
setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 2-3 FSCV RMSE")
ggsave( "multiTaskSuppSize_fscv_noLasso.pdf",
        plot = plt_supp,
        width = 15,
        height = 7
)
