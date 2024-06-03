#######################
# compare different K 
#######################
library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)

Kvec <- c(8, 6, 4) # number of tasks
pVec <- c(100, 500, 1000) # dimension of covariates
sVec <- c(5,10,25,50) # cardinality constraint
itrs <- 180 # number of simulation iterations
cnt <- 0

for(rho in sVec){
  setwd("~/Desktop/Research/bcancer")
  ls_supp <- ls_beta <- vector(length = length(Kvec), "list")
  for(p in pVec){
    for(k in Kvec){
        flNm <-  paste0("MT_breast_rand__K_", k, "_study_17_p_", p, "_totSims_180_L0sseTn_sse_rdgLmd_TRUE_MSTn_multiTask_Yscl_TRUE_rhoMax_", rho, "_nFld_5_LSitr_50_LSspc_1_fitLocal_50_Zlen_78_wsMeth_2_asPass_TRUE_TnIn_TRUE")
                        # "MT_breast_rand__K_4_study_17_p_1000_totSims_180_L0sseTn_sse_rdgLmd_TRUE_MSTn_multiTask_Yscl_TRUE_rhoMax_25_nFld_5_LSitr_50_LSspc_1_fitLocal_50_Zlen_78_wsMeth_2_asPass_TRUE_TnIn_TRUE"
        if(file.exists(flNm)){
            # check to see if file exists
            cnt <- cnt + 1
            d <- read.csv(flNm) 
            
            # remove NAs
            ind <- apply(d, 1, function(x) all(is.na(x)))
            d <- d[!ind,]
            
            rmseMat <- data.frame(
                                  #"mrgL0" = d$mrgL0  / d$MT_oseL0,
                                  "CS+Bbar" = d$MT_msP1_L0 / d$MT_oseL0,
                                  "Zbar+L2" = d$MT_msP2_L0 / d$MT_oseL0,
                                  "CS+L2" = d$MT_msP3_L0 / d$MT_oseL0,
                                  "Bbar" = d$MT_msP4_L0 / d$MT_oseL0,
                                  "Zbar+Bbar" = d$MT_msP5_L0 / d$MT_oseL0,
                                  #"CVX_Bbar" = d$MT_msP1_con / d$MT_oseL0,
                                  #"Ridge" = d$MT_msP3_con / d$MT_oseL0,
                                  #"Trace" = d$MTL_trace / d$MT_oseL0,
                                  "LASSO"= d$MT_lasso_low_s / d$MT_oseL0,
                                  "SGL" = d$SGL_MT / d$MT_oseL0,
                                  "GL" = d$GL_MT / d$MT_oseL0,
                                  "grMCP" = d$grMCP_MT / d$MT_oseL0,
                                  "gel" = d$gel_MT / d$MT_oseL0,
                                  "cMCP" = d$cMCP_MT / d$MT_oseL0 )
            
            rmseMat <- rmseMat[complete.cases(rmseMat), ] # include if observations for all so comparisons are fair
            names(rmseMat) <- gsub(x = names(rmseMat), pattern = "\\.", replacement = "+")
            
            suppMat <- data.frame(
                                  "CS+Bbar" = d$s_MS1 - d$s_ose,
                                  "Zbar+L2" = d$s_MS2 - d$s_ose,
                                  "CS+L2" = d$s_MS3 - d$s_ose,
                                  "Bbar" = d$s_MS4 - d$s_ose,
                                  "Zbar+Bbar" = d$s_MS5 - d$s_ose,
                                  "LASSO"= d$MT_lasso_low - d$s_ose)
            
            names(suppMat) <- gsub(x = names(suppMat), pattern = "\\.", replacement = "+")
            suppMat <- suppMat[complete.cases(suppMat), ] # include if observations for all
            
            ls_beta[[cnt]] <- cbind( 
                gather(rmseMat), 
                k, p)
            
            ls_supp[[cnt]] <- cbind( 
                gather(suppMat), 
                k, p)
            
            d1 <- d
            
            rm(d)
          }
      }
  }


dat <- do.call(rbind, ls_beta)
dat$k <- as.factor(dat$k)
dat$p <- as.factor(dat$p)
####################################################################################################
# set colors
# group 1
levs <- grp_incl <- c("Bbar", "SGL", "Zbar+L2", "gel", "grMCP", "CS+L2")
myColors <- setNames( c("#ca0020", "darkgrey", "#0868ac", "#525252", "#E69F00", "darkgreen"), levs) 
# grp_incl <- c("Bbar", "SGL", "Zbar+L2", "grMCP", "gel", "cMCP")

# group 2 - for appendix
grp_incl2 <- levs2 <- c("GL", "CS+L2", "CS+Bbar", "Zbar+L2", "Zbar+Bbar", "cMCP")
myColors2 <- setNames( c("#ca0020", "darkgrey", "#0868ac", "#525252", "#E69F00", "darkgreen"), levs2) 
# grp_incl2 <- c("Bbar", "GL", "CS+Bbar", "CS+L2", "Zbar+Bbar")


#########################
# rmse -- only L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
    dat %>% tibble %>%  
    dplyr::filter(key %in% levs) %>%
    ggplot(aes( y = value, x = p, fill = key )) +
   facet_wrap( ~ k, nrow = 1) +
    geom_boxplot(
        lwd = 1, 
        fatten = 0.5) + 
    geom_hline(yintercept=1, 
               linetype="dashed", 
               color = "black", 
               size = rel(0.5),
               alpha = 0.7) + #
    ylab(TeX('${log(RMSE_{Method}/RMSE_{L_0 L_2})}$') )+ 
    xlab("Number of Features (p)") + 
    scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
    scale_color_manual(values = myColors[levs %in% grp_incl] ) +
    theme_classic(base_size = 12) +
    coord_cartesian(ylim = c(0.6, 1.2) ) + 
    theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
           axis.text=element_text(face="bold",color="black", size=rel(2)),
           axis.title = element_text(face="bold", color="black", size=rel(2)),
           legend.key.size = unit(2, "line"), # added in to increase size
           legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
           legend.title = element_text(face="bold", color="black", size = rel(2)),
           strip.text.x = element_text(face="bold", color="black", size = rel(2.5))) + 
    guides(fill= guide_legend(title="Method")) 

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer/Resubmission /Figures")
ggsave( paste0("multiTaskRMSE_bCancer_L0_", rho ,".pdf"),
        plot = plt_rmse,
        width = 15,
        height = 7)

#########################
# rmse -- non-L0 methods
#########################
# common support vs. mrg and ose
plt_rmse = 
  dat %>% tibble %>%  
  dplyr::filter(key %in% levs2) %>%
  ggplot(aes( y = value, x = p, fill = key )) +
  facet_wrap( ~ k, nrow = 1) +
  geom_boxplot(
    lwd = 1, 
    fatten = 0.5) + 
  geom_hline(yintercept=1, 
             linetype="dashed", 
             color = "black", 
             size = rel(0.5),
             alpha = 0.7) + #
  ylab(TeX('${log(RMSE_{Method}/RMSE_{L_0 L_2})}$') )+ 
  xlab("Number of Features (p)") + 
  scale_fill_manual(values = myColors2[levs2 %in% grp_incl2] ) +
  scale_color_manual(values = myColors2[levs2 %in% grp_incl2] ) +
  theme_classic(base_size = 12) +
  coord_cartesian(ylim = c(0.6, 1.2) ) + 
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(2)),
         axis.title = element_text(face="bold", color="black", size=rel(2)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(2)),
         strip.text.x = element_text(face="bold", color="black", size = rel(2.5))) + 
  guides(fill= guide_legend(title="Method")) 

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer/Resubmission /Figures")
ggsave( paste0("multiTaskRMSE_bCancer_all_", rho ,".pdf"),
        plot = plt_rmse,
        width = 15,
        height = 7)

###########################################
}











# 
# 
# ###########################################
# dat <- do.call(rbind, ls_supp)
# dat$k <- as.factor(dat$k)
# 
# 
# #########################
# # supp
# #########################
# # common support vs. mrg and ose
# plt_supp = 
#     dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
#     dplyr::filter(
#                   key %in% c("CS+L2", "Zbar+Bbar","Bbar", "Zbar+L2", "LASSO"), # , "LASSO", "Trace"
#     ) %>%
#     dplyr::group_by(key, k) %>% 
#     dplyr::summarize(my_mean = mean(value) ) %>% 
#     arrange(k) %>% #print(n = Inf)
#     ggplot(aes(y = my_mean, x = k, fill = key)) +
#     # y = value, x = e1, fill = key 
#     #facet_wrap( ~ n, nrow = 1) +
#     geom_bar(stat="identity", position=position_dodge()) +
#     # coord_cartesian(ylim = c(-0.1, 0.5) ) + 
#     ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
#     xlab("K") + 
#     scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
#     scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
#     theme_classic(base_size = 12) +
#     #ylim(0.8, 1.1) + 
#     theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
#            axis.text=element_text(face="bold",color="black", size=rel(2)),
#            axis.title = element_text(face="bold", color="black", size=rel(2)),
#            legend.key.size = unit(2, "line"), # added in to increase size
#            legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
#            legend.title = element_text(face="bold", color="black", size = rel(2)),
#            strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
#     ) + guides(fill= guide_legend(title="Method"))#guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))
# 
# setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
# ggsave( "multiTaskSuppSize_bCancer.pdf",
#         plot = plt_supp,
#         width = 15,
#         height = 7
# )
# 
# 
# plt_supp = 
#     dat %>% tibble %>%  # "CS+Bbar", "CS+L2", , "Zbar+Bbar"
#     dplyr::filter(
#         key %in% c("CS+L2", "Zbar+Bbar","Bbar", "Zbar+L2"), # , "LASSO", "Trace"
#     ) %>%
#     dplyr::group_by(key, k) %>% 
#     dplyr::summarize(my_mean = mean(value) ) %>% 
#     arrange(k) %>% #print(n = Inf)
#     ggplot(aes(y = my_mean, x = k, fill = key)) +
#     # y = value, x = e1, fill = key 
#     #facet_wrap( ~ n, nrow = 1) +
#     geom_bar(stat="identity", position=position_dodge()) +
#     # coord_cartesian(ylim = c(-0.1, 0.5) ) + 
#     ylab(TeX('$\\mathbf{\\rho_{Method} - \\rho_{\u2113_0}}$') )+ 
#     xlab("K") + 
#     scale_fill_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray") ) +
#     scale_color_manual(values = c("#ca0020", "lightgrey", "#0868ac", "#E69F00", "#525252", "darkgray")) +
#     theme_classic(base_size = 12) +
#     #ylim(0.8, 1.1) + 
#     theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(2.5), face="bold"),
#            axis.text=element_text(face="bold",color="black", size=rel(2)),
#            axis.title = element_text(face="bold", color="black", size=rel(2)),
#            legend.key.size = unit(2, "line"), # added in to increase size
#            legend.text = element_text(face="bold", color="black", size = rel(2)), # 3 GCL
#            legend.title = element_text(face="bold", color="black", size = rel(2)),
#            strip.text.x = element_text(face="bold", color="black", size = rel(2.5))
#     ) + guides(fill= guide_legend(title="Method"))#guides(fill=guide_legend(title=TeX('$\\mathbf{\\sigma^2_{x}}$')))
# 
# setwd("~/Desktop/Research Final/Sparse Multi-Study/Figures/Figure 3 - Breast Cancer")
# ggsave( "multiTaskSuppSize_bCancer_noLasso.pdf",
#         plot = plt_supp,
#         width = 15,
#         height = 7
# )
# 
