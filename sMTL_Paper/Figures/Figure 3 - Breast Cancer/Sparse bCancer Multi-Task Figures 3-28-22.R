#######################
# compare different K 
#######################
setwd("~/Desktop/Research/breastCancer4")
library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)

Kvec <- c(2, 4, 6)
ls_supp <- ls_beta <- vector(length = length(Kvec), "list")

itrs <- 18 # number of simulation iterations
cnt <- 0
rmseMat <- matrix(nc = 4, nr = itrs)

for(k in Kvec){
    flNm <-  paste0("MT_breast_K_", k, "_study_17_ds_1_totSims_90_L0sseTn_sse_rdgLmd_TRUE_MSTn_multiTask_Yscl_TRUE_rhoMax_50_nFld_3_LSitr_50_LSspc_1_fitLocal_50_Zlen_19_wsMeth_2_asPass_TRUE_TnIn_TRUE")

    if(file.exists(flNm)){
        # check to see if file exists
        cnt <- cnt + 1
        d <- read.csv(flNm) 
        
        # remove NAs
        ind <- apply(d, 1, function(x) all(is.na(x)))
        d <- d[!ind,]
        
        rmseMat <- data.frame("LASSO" = d$MT_mtLasso / d$MT_oseL0,
                              "CS+Bbar" = d$MT_msP1_L0 / d$MT_oseL0,
                              "Zbar+L2" = d$MT_msP2_L0 / d$MT_oseL0,
                              "CS+L2" = d$MT_msP3_L0 / d$MT_oseL0,
                              "Bbar" = d$MT_msP4_L0 / d$MT_oseL0,
                              "Zbar+Bbar" = d$MT_msP5_L0 / d$MT_oseL0,
                              "CVX_Bbar" = d$MT_msP1_con / d$MT_oseL0,
                              "Ridge" = d$MT_msP3_con / d$MT_oseL0,
                              "Trace" = d$MTL_trace / d$MT_oseL0,
                              "LASSO_low"= d$MT_lasso_low
        )
        
        rmseMat <- rmseMat[complete.cases(rmseMat), ] # include if observations for all so comparisons are fair
        names(rmseMat) <- gsub(x = names(rmseMat), pattern = "\\.", replacement = "+")
        
        suppMat <- data.frame(
            "CS+Bbar" = d$s_MS1 - d$s_ose,
            "Zbar+L2" = d$s_MS2 - d$s_ose,
            "CS+L2" = d$s_MS3 - d$s_ose,
            "Bbar" = d$s_MS4 - d$s_ose,
            "Zbar+Bbar" = d$s_MS5 - d$s_ose,
            "LASSO" = d$s_lassoMrg - d$s_ose,
            "LASSO_low"= d$MT_lasso_low_s - d$s_ose
            
        )
        
        names(suppMat) <- gsub(x = names(suppMat), pattern = "\\.", replacement = "+")
        suppMat <- suppMat[complete.cases(suppMat), ] # include if observations for all
        
        
        ls_beta[[cnt]] <- cbind( 
            gather(rmseMat), 
            k
        )
        
        ls_supp[[cnt]] <- cbind( 
            gather(suppMat), 
            k
        )
        
        d1 <- d
        
        rm(d)
    }
    
}



dat <- do.call(rbind, ls_beta)
dat$k <- as.factor(dat$k)

#########################
# rmse -- only L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
                  key %in% c("CS+Bbar", "CS+L2", "Zbar+Bbar","Bbar", "Zbar+L2"), # , "LASSO", "Trace"
                   ) %>%
    ggplot(aes( y = value, x = k, fill = key )) +
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
    #ylim(0, 2) +
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') )+ 
    xlab("K") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.6, 1.2) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  #guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
ggsave( "multiTaskRMSE_bCancer_L0.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)

#########################
# rmse -- non-L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
                  key %in% c("Zbar+L2", "LASSO", "Trace", "Ridge"), # , 
    ) %>%
    ggplot(aes( y = value, x = k, fill = key )) +
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
    #ylim(0, 2) +
    ylab(TeX('$\\mathbf{RMSE_{Method}/RMSE_{\u2113_0}}$') )+ 
    xlab("K") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.6, 1.25) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))  #guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
ggsave( "multiTaskRMSE_bCancer_all.pdf",
        plot = plt_rmse,
        width = 15,
        height = 7
)

###########################################

###########################################
dat <- do.call(rbind, ls_supp)
dat$k <- as.factor(dat$k)


#########################
# supp
#########################
# common support vs. mrg and ose
plt_supp = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
                  key %in% c("CS+L2", "Zbar+Bbar","Bbar", "Zbar+L2", "LASSO"), # , "LASSO", "Trace"
    ) %>%
    dplyr::group_by(key, k) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(k) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = k, fill = key)) +
    # y = value, x = e1, fill = key 
    #facet_wrap( ~ n, nrow = 1) +
    geom_bar(stat="identity", position=position_dodge()) +
    # coord_cartesian(ylim = c(-0.1, 0.5) ) + 
    ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
    xlab("K") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    #ylim(0.8, 1.1) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))#guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
ggsave( "multiTaskSuppSize_bCancer.pdf",
        plot = plt_supp,
        width = 15,
        height = 7
)


plt_supp = 
    dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
    dplyr::filter(
        key %in% c("CS+L2", "Zbar+Bbar","Bbar", "Zbar+L2"), # , "LASSO", "Trace"
    ) %>%
    dplyr::group_by(key, k) %>% 
    dplyr::summarize(my_mean = mean(value) ) %>% 
    arrange(k) %>% #print(n = Inf)
    ggplot(aes(y = my_mean, x = k, fill = key)) +
    geom_bar(stat="identity", position=position_dodge()) +
    ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
    xlab("K") + 
    scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
    scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
    theme_classic(base_size = 12) +
    #ylim(0.8, 1.1) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
    ) + guides(fill= guide_legend(title="Method"))#guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))

setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
ggsave( "multiTaskSuppSize_bCancer_noLasso.pdf",
        plot = plt_supp,
        width = 15,
        height = 7
)

