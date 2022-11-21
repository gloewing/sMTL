# Figures and Tables
f1score <- function(data, name){
    tp <- data[ paste0(name, "_tp") ]
    fp <- data[ paste0(name, "_fp") ]
    
    return( tp / (tp + 0.5 * (fp + 1 - tp ) ) ) # f1 score
}

##########################################
# compare different K and betaVar -- no regularization
######################################################
setwd("~/Desktop/Research/sparseMT6")
library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)
bVar <- c(10, 50)
xVar <- c(0)  
Kvec <- c(2,4,6,8)
ls_f1 <- ls_supp <- ls_tp <- ls_fp <- ls_beta <- ls_coef <- vector(length = length(bVar) * length(Kvec), "list")

bMean <- "0.2_0.5"
epsVecUp <- c(5, 0.5, 0.05, 0.005)
epsVecLow <- c(20, 2, 0.2, 0.02)
nVec <- c(50,100,200,400)
rVec <- c(5,10,15,20)
sVec <- c(10,20,30)
pVec <- c(250, 1000)

itrs <- 100 # number of simulation iterations
cnt <- 0
rmseMat <- matrix(nc = 4, nr = itrs)
for(p in pVec){
for(r in rVec){
    for(s in sVec){
        for(n in nVec){
            for(k in Kvec){
                for(bl in 1:length(bVar)){
                    for(ep in 1:length(epsVecLow)){
                        b <- bVar[bl]
                        e2 <- epsVecLow[ep]
                        e1 <- epsVecUp[ep]
    
                        flNm <-  paste0("sprsMS_LS_s_",s,"_r_",r,"_rp_0.5_numCovs_250_n_",n, ".", n, "_eps_", e1, ".", e2, "_covTyp_exponential_rho_0.5_clustDiv_10_bVar_", b, "_xVar_0_clst_", k, "_K_", k, "_bMean_0.2_0.5__bFix_TRUE_L0sseTn_sse_MSTn_multiTask_nFld_10_LSitr_50_LSspc_1_wsMeth_2_asPass_TRUE_TnIn_TRUEcat_FALSE")
                                        # sprsMS_LS_s_10_r_5_rp_0.5_numCovs_250_n_50.50_eps_0.05.0.2_covTyp_exponential_rho_0.5_clustDiv_10_bVar_10_xVar_0_clst_2_K_2_bMean_0.2_0.5__bFix_TRUE_L0sseTn_sse_MSTn_multiTask_nFld_10_LSitr_50_LSspc_1_wsMeth_2_asPass_TRUE_TnIn_TRUEcat_FALSE
                        if(file.exists(flNm)){
                            # check to see if file exists
                            cnt <- cnt + 1
                            d <- read.csv(flNm) 
                            
                            # remove NAs
                            ind <- apply(d, 1, function(x) all(is.na(x)))
                            d <- d[!ind,]
                            
                            rmseMat <- data.frame("LASSO" = d$MT_mrgLasso / d$MT_oseL0,
                                                  "Ridge" = d$MT_mrgRdg / d$MT_oseL0,
                                                  "mrgL0" = d$mrgL0  / d$MT_oseL0,
                                                  "CS+Bbar" = d$MT_msP1_L0 / d$MT_oseL0,
                                                  "Zbar+L2" = d$MT_msP2_L0 / d$MT_oseL0,
                                                  "CS+L2" = d$MT_msP3_L0 / d$MT_oseL0,
                                                  "Bbar" = d$MT_msP4_L0 / d$MT_oseL0,
                                                  "Zbar+Bbar" = d$MT_msP5_L0 / d$MT_oseL0,
                                                  "ms1_cvx" = d$MT_msP1_con / d$MT_oseL0,
                                                  "ms3_cvx" = d$MT_msP3_con / d$MT_oseL0,
                                                  "Trace" = d$MTL_trace / d$MT_oseL0
                            )
                            
                            names(rmseMat) <- gsub(x = names(rmseMat), pattern = "\\.", replacement = "+")
    
                            
                            suppMat <- data.frame("mrgL0" = d$mrgL0_sup - d$oseL0_sup,
                                                  "CS+Bbar" = d$msP1_L0_sup - d$oseL0_sup,
                                                  "Zbar+L2" = d$msP2_L0_sup - d$oseL0_sup,
                                                  "CS+L2" = d$msP3_L0_sup - d$oseL0_sup,
                                                  "Bbar" = d$msP4_sup - d$oseL0_sup,
                                                  "Zbar+Bbar" = d$msP5_sup - d$oseL0_sup,
                                                  "LASSO" = d$lassoMrg_sup - d$oseL0_sup
                                                  
                            )
                            
                            names(suppMat) <- gsub(x = names(suppMat), pattern = "\\.", replacement = "+")
                            
                            
                            coefMat <- data.frame("mrg_L0" = d$coef_L0Mrg / d$coef_oseL0,
                                                  "mrg_rdg" = d$coef_mrgRdg / d$coef_oseL0,
                                                  "CS+Bbar" = d$coef_msP1_L0 / d$coef_oseL0,
                                                  "Zbar+L2" = d$coef_msP2_L0 / d$coef_oseL0,
                                                  "CS+L2" = d$coef_msP3_L0 / d$coef_oseL0,
                                                  "Bbar" = d$coef_msP4_L0 / d$coef_oseL0,
                                                  "Zbar+Bbar" = d$coef_msP5_L0 / d$coef_oseL0,
                                                  "ms1_cvx" = d$coef_msP1_con / d$coef_oseL0,
                                                  "ms3_cvx" = d$coef_msP3_con / d$coef_oseL0
                            )
                            
                            names(coefMat) <- gsub(x = names(coefMat), pattern = "\\.", replacement = "+")
                            
                            f1 <- data.frame("mrg_L0" = f1score(d, "mrgL0") / f1score(d, "oseL0"),
                                                  "CS+Bbar" = f1score(d, "msP1_L0") / f1score(d, "oseL0"),
                                                  "Zbar+L2" = f1score(d, "msP2_L0") / f1score(d, "oseL0"),
                                                  "CS+L2" = f1score(d, "msP3_L0") / f1score(d, "oseL0"),
                                                  "Bbar" = f1score(d, "msP4") / f1score(d, "oseL0"),
                                                  "Zbar+Bbar" = f1score(d, "msP5") / f1score(d, "oseL0"),
                                             "LASSO" = f1score(d, "lassoMrg") / f1score(d, "oseL0")
                            )
                            
                            # make sure the names line up with suppMat!!
                            names(f1) <- names(suppMat)#gsub(x = names(f1), pattern = "\\.", replacement = "+")
                            
                            tpMat <- data.frame(mrgL0 = d$mrgL0_tp,
                                                oseL0 = d$oseL0_tp,
                                                ms1 = d$msP1_L0_tp,
                                                ms2 = d$msP2_L0_tp,
                                                ms3 = d$msP3_L0_tp,
                                                mrgLasso = d$lassoMrg_tp
                                                
                            )
                            
                            fpMat <- data.frame(mrgL0 = d$mrgL0_fp,
                                                oseL0 = d$oseL0_fp,
                                                ms1 = d$msP1_L0_fp,
                                                ms2 = d$msP2_L0_fp,
                                                ms3 = d$msP3_L0_fp,
                                                mrgLasso = d$lassoMrg_fp
                            )
                            
                            
                            ls_beta[[cnt]] <- cbind( 
                                gather(rmseMat), 
                                b,k,n,e1,r,s,p
                            )
                            
                            ls_supp[[cnt]] <- cbind( 
                                gather(suppMat), 
                                b,k,n, e1,r,s,p
                            )
                            
                            ls_fp[[cnt]] <- cbind( 
                                gather(fpMat), 
                                b,k,n, e1,r,s,p
                            )
                            
                            ls_tp[[cnt]] <- cbind( 
                                gather(tpMat), 
                                b,k,n,e1,r,s,p
                            )
                            
                            ls_coef[[cnt]] <- cbind( 
                                gather(coefMat), 
                                b,k,n,e1,r,s,p
                            )
                            
                            ls_f1[[cnt]] <- cbind( 
                                gather(f1), 
                                b,k,n,e1,r,s,p
                            )
                            
                            d1 <- d
                            rm(d)
                        }
                        
                        
                        
                    }
                }
            }
        }
    }
}
}

# factors
dat <- do.call(rbind, ls_beta)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r <- as.factor(dat$r)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)

#########################
# rmse
#########################
plt_rmse = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  b == 50,
                  r == 5,
                  n == 50,
                  e1 != 0.01,
                  p == 1000,
                  !k %in% c(8) ) %>%
    ggplot(aes( y = value, x = e1, fill = key )) +
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
    # ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') )+ 
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{TS-SR}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0, 1.1) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskRMSE.pdf",
        plot = plt_rmse,
        width = 12,
        height = 6
)
###########################################
dat <- do.call(rbind, ls_coef)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# coef
#########################
plt_coef = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  n == 50,
                  e1 != 0.01,
                  p == 1000,
                  !k %in% c(4, 8) ) %>%
    ggplot(aes( y = value, x = e1, fill = key )) +
    facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(0, 1.3) ) + 
    # ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') ) + 
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{TS-SR}$') ) + 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskCoef.pdf",
        plot = plt_coef,
        width = 12,
        height = 6
)

###########################################
###########################################
dat <- do.call(rbind, ls_supp)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# supp
#########################
plt_supp = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  n == 50,
                  e1 != 0.01,
                  p == 1000,
                  !k %in% c(8) ) %>%
    dplyr::group_by(key, b, k, e1) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(b, k, e1) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = e1, fill = key)) +
    facet_wrap( ~ k, nrow = 1) +
    geom_bar(stat="identity", position=position_dodge()) +
    coord_cartesian(ylim = c(-0.1, 0.5) ) + 
    ylab(TeX('$\\mathbf{Supp_{Method} - Supp_{TS-SR}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskSupp.pdf",
        plot = plt_supp,
        width = 12,
        height = 6
)

############################
# f1
dat <- do.call(rbind, ls_f1)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)
dat %>% tibble %>% 
    dplyr::group_by(key, b,k,n, e1) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% print(n = Inf)

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# f1
#########################
plt_f1 = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  n == 50,
                  b == 10,
                  e1 != 0.01,
                  p == 1000,
                  !k %in% c(8) ) %>%
    ggplot(aes( y = value, x = e1, fill = key )) +
    facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(1.0, 2.25) ) + 
    ylab(TeX('$\\mathbf{F1_{Method}/F1_{TS-SR}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), 
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + 
    guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskF1.pdf",
        plot = plt_f1,
        width = 12,
        height = 6
)
#############################################################################
########################################
# as a function of sample size
########################################

#########################
# rmse
#########################

dat <- do.call(rbind, ls_beta)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r <- as.factor(dat$r)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  e1 == 10,
                  k %in% c(8) ) %>%
    ggplot(aes( y = value, x = n, fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0, 1.1) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskRMSE_n.pdf",
        plot = plt_rmse,
        width = 12,
        height = 6
)
###########################################
###########################################
dat <- do.call(rbind, ls_coef)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)
dat %>% tibble %>% 
    dplyr::group_by(key, b,k,n, e1) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% print(n = Inf)

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# coef
#########################
plt_coef = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  e1 == 10,
                  k %in% c(8) ) %>%
    ggplot(aes( y = value, x = n, fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(0, 1.3) ) + 
    ylab(TeX('$\\mathbf{Coef_{Method}/Coef_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskCoef_n.pdf",
        plot = plt_coef,
        width = 12,
        height = 6
)

###########################################
###########################################
dat <- do.call(rbind, ls_supp)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# supp
#########################
plt_supp = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  e1 == 10,
                  k %in% c(8) ) %>%
    dplyr::group_by(key, b, k, e1) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(b, k, e1) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = n, fill = key)) +
    geom_bar(stat="identity", position=position_dodge()) +
    coord_cartesian(ylim = c(-0.1, 0.5) ) + 
    ylab(TeX('$\\mathbf{Supp_{Method} - Supp_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskSupp_n.pdf",
        plot = plt_supp,
        width = 12,
        height = 6
)

############################
# f1
dat <- do.call(rbind, ls_f1)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)


setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
#########################
# f1
#########################
plt_f1 = 
    dat %>% tibble %>%  
    dplyr::filter(s == 10, 
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
                  r == 5,
                  e1 == 10,
                  k %in% c(8) ) %>%
    ggplot(aes( y = value, x = n, fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(1.0, 2.25) ) + 
    ylab(TeX('$\\mathbf{F1_{Method}/F1_{\u2113_0}}$') )+ 
    xlab(TeX('$\\mathbf{\\tau}$')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskF1_n.pdf",
        plot = plt_f1,
        width = 12,
        height = 6
)
#############################################################################

#############################################################################
########################################
# as a function of r and s
########################################

#########################
# rmse - r & s
#########################

dat <- do.call(rbind, ls_beta)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

plt_rmse = 
    dat %>% tibble %>%  
    dplyr::filter(
        key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
        n == 50,
                  as.numeric(r) / as.numeric(s) == 0.5,
                  e1 == 10,
                  k %in% c(6) ) %>%
    ggplot(aes( y = value, x = as.factor(s), fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') )+ 
    xlab(TeX('\\mathbf{Fixed Support}')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0, 1.1) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskRMSE_rs.pdf",
        plot = plt_rmse,
        width = 12,
        height = 6
)

###########################################


#########################
# coef
#########################
dat <- do.call(rbind, ls_coef)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

plt_coef = 
    dat %>% tibble %>%  
    dplyr::filter(
        key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
        n == 50,
        as.numeric(r) / as.numeric(s) == 0.5,
        e1 == 10,
        k %in% c(6) ) %>%
    ggplot(aes( y = value, x = as.factor(s), fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(0, 1.3) ) + 
    ylab(TeX('$\\mathbf{Coef_{Method}/Coef_{\u2113_0}}$') )+ 
    xlab(TeX('\\mathbf{Fixed Support}')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskCoef_rs.pdf",
        plot = plt_coef,
        width = 12,
        height = 6
)

###########################################
###########################################
dat <- do.call(rbind, ls_supp)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

plt_supp = 
    dat %>% tibble %>%  
    dplyr::filter(
        key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
        n == 50,
        as.numeric(r) / as.numeric(s) == 0.5,
        e1 == 10,
        k %in% c(6) ) %>%
    dplyr::group_by(key, b, k, e1) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(b, k, e1) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = n, fill = key)) +
    geom_bar(stat="identity", position=position_dodge()) +
    coord_cartesian(ylim = c(-0.1, 0.5) ) + 
    ylab(TeX('$\\mathbf{Supp_{Method} - Supp_{\u2113_0}}$') )+ 
    xlab(TeX('\\mathbf{Fixed Support}')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskSupp_rs.pdf",
        plot = plt_supp,
        width = 12,
        height = 6
)

############################
# f1
dat <- do.call(rbind, ls_f1)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$e1 <- as.factor(2 * dat$e1)

plt_f1 = 
    dat %>% tibble %>%  
    dplyr::filter(
        key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "LASSO", "Zbar+L2"),
        n == 50,
        as.numeric(r) / as.numeric(s) == 0.5,
        e1 == 10,
        k %in% c(6) ) %>%
    ggplot(aes( y = value, x = as.factor(s), fill = key )) +
    geom_boxplot(
        lwd = 1.0, 
        fatten = 0.5
    ) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    coord_cartesian(ylim = c(1.0, 2.25) ) + 
    ylab(TeX('$\\mathbf{F1_{Method}/F1_{\u2113_0}}$') )+ 
    xlab(TeX('\\mathbf{Fixed Support}')) + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "darkgray", "#E69F00", "#525252")) +
    theme_classic(base_size = 12) +
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 5 Sims")
ggsave( "multiTaskF1_rs.pdf",
        plot = plt_f1,
        width = 12,
        height = 6
)
#############################################################################
