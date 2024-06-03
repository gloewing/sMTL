# no support overlap
# Figures and Tables
f1score <- function(data, name){
    tp <- data[ paste0(name, "_tp") ]
    fp <- data[ paste0(name, "_fp") ]
    
    return( tp / (tp + 0.5 * (fp + 1 - tp ) ) ) # f1 score
}

#######################################################
# compare different K and betaVar -- no regularization
#######################################################
setwd("/Users/loewingergc/Desktop/Research/smtl_sims0")
library(dplyr)
library(latex2exp)
library(kableExtra)
library(tidyverse)
library(ggpubr)

bVar <- c(10, 50)
xVar <- c(0)  
Kvec <- c(4)
ls_s <- ls_f1 <- ls_supp <- ls_tp <- ls_fp <- ls_beta <- ls_beta2 <- ls_coef <- vector(length = length(bVar) * length(Kvec), "list")

bMean <- "0.2_0.5"
epsVecUp <- c(25, 5, 0.5, 0.05, 0.005)
epsVecLow <- c(100, 20, 2, 0.2, 0.02)
nVec <- c(50,100,150)
rVec <- c(0.2,0.5,0.8) # 20 # vector of exponential correlation rhos
s <- 0
p <- 250
r <- 20
qVec <- 0 # no support overlap
mixProb <- c("NA")# c(3/4)
mixSD <- c("NA")#c(5)
rhoVec <- c(7, 10, 13)
itrs <- 100 # number of simulation iterations
cnt <- 0
rmseMat <- matrix(nc = 4, nr = itrs)
for(q in qVec){
  for(mix in 1:length(mixSD)){
    for(r_exp in rVec){
        for(rho in rhoVec){
            for(n in nVec){
                for(k in Kvec){
                    for(bl in 1:length(bVar)){
                        for(ep in 1:length(epsVecLow)){
                            b <- bVar[bl]
                            e2 <- epsVecLow[ep]
                            e1 <- epsVecUp[ep]
        
                            flNm <-  paste0("MTL_s_",s,"_r_",r,"_rp_0.5_q_", q,"_p_250_n_",n, ".", n, "_eps_", e1, ".", e2, "_covTyp_exponential_rho_", r_exp, "_cDv_10_bVar_", b, "_xVar_0_K_", k, "_bMu_0.2_0.5__bFx_TRUE_sseTn_sse_MSTn_multiTask_nFld_10_LSitr_50_LSspc_1_wsM_1_asP_TRUE_TnIn_TRUEcat_4_mxP_", mixProb[mix], "_mxSD_", mixSD[mix], "_rho_", rho)
                                         #   "MTL_s_0_r_20_rp_0.5_q_10_p_250_n_50.50_eps_0.5.2_covTyp_exponential_rho_0.5_cDv_10_bVar_50_xVar_0_K_4_bMu_0.2_0.5__bFx_TRUE_sseTn_sse_MSTn_multiTask_nFld_10_LSitr_50_LSspc_1_wsM_1_asP_TRUE_TnIn_TRUEcat_4_mxP_0.75_mxSD_5_rho_7"
                            if(file.exists(flNm)){
                                # check to see if file exists
                                cnt <- cnt + 1
                                d <- read.csv(flNm) 
                                mm <- ifelse(is.na(mixSD[mix]), 0, mixSD[mix])
                                
                                # remove NAs
                                ind <- apply(d, 1, function(x) all(is.na(x)))
                                d <- d[!ind,]

                                rmseMat <- data.frame("Ridge" = d$MT_mrgRdg / d$MT_oseL0,
                                                      "mrgL0" = d$mrgL0  / d$MT_oseL0,
                                                      "CS+Bbar" = d$MT_msP1_L0 / d$MT_oseL0,
                                                      "Zbar+L2" = d$MT_msP2_L0 / d$MT_oseL0,
                                                      "CS+L2" = d$MT_msP3_L0 / d$MT_oseL0,
                                                      "Bbar" = d$MT_msP4_L0 / d$MT_oseL0,
                                                      "Zbar+Bbar" = d$MT_msP5_L0 / d$MT_oseL0,
                                                      "ms1_cvx" = d$MT_msP1_con / d$MT_oseL0,
                                                      "ms3_cvx" = d$MT_msP3_con / d$MT_oseL0,
                                                      "Trace" = d$MTL_trace / d$MT_oseL0,
                                                      "SGL" = d$SGL_MT / d$MT_oseL0,
                                                      "GL" = d$GL_MT / d$MT_oseL0,
                                                      "grMCP" = d$grMCP_MT / d$MT_oseL0,
                                                      "gel" = d$gel_MT / d$MT_oseL0,
                                                      "cMCP" = d$cMCP_MT / d$MT_oseL0,
                                                      "Lasso" = d$lasso_MT / d$MT_oseL0)
                                
                                names(rmseMat) <- gsub(x = names(rmseMat), pattern = "\\.", replacement = "+")
                                
                                rmseMat2 <- data.frame("Ridge" = d$MT_mrgRdg,
                                                      "mrgL0" = d$mrgL0 ,
                                                      "CS+Bbar" = d$MT_msP1_L0,
                                                      "Zbar+L2" = d$MT_msP2_L0,
                                                      "CS+L2" = d$MT_msP3_L0,
                                                      "Bbar" = d$MT_msP4_L0,
                                                      "Zbar+Bbar" = d$MT_msP5_L0,
                                                      "ms1_cvx" = d$MT_msP1_con,
                                                      "ms3_cvx" = d$MT_msP3_con,
                                                      "Trace" = d$MTL_trace,
                                                      "SGL" = d$SGL_MT,
                                                      "GL" = d$GL_MT,
                                                      "grMCP" = d$grMCP_MT,
                                                      "gel" = d$gel_MT,
                                                      "cMCP" = d$cMCP_MT,
                                                      "Lasso" = d$lasso_MT)
                                
                                names(rmseMat2) <- gsub(x = names(rmseMat2), pattern = "\\.", replacement = "+")
        
                                
                                suppMat <- data.frame("mrgL0" = d$mrgL0_sup - d$oseL0_sup,
                                                      "CS+Bbar" = d$msP1_L0_sup - d$oseL0_sup,
                                                      "Zbar+L2" = d$msP2_L0_sup - d$oseL0_sup,
                                                      "CS+L2" = d$msP3_L0_sup - d$oseL0_sup,
                                                      "Bbar" = d$msP4_sup - d$oseL0_sup,
                                                      "Zbar+Bbar" = d$msP5_sup - d$oseL0_sup,
                                                      "Trace" = d$MTL_trace - d$oseL0_sup,
                                                      "SGL" = d$SGL_MT - d$oseL0_sup,
                                                      "GL" = d$GL_sup - d$oseL0_sup,
                                                      "grMCP" = d$grMCP_sup - d$oseL0_sup,
                                                      "gel" = d$gel_sup - d$oseL0_sup,
                                                      "cMCP" = d$cMCP_sup - d$oseL0_sup,
                                                      "Lasso" = d$lasso_sup - d$oseL0_sup)
                                                      
                                
                                names(suppMat) <- gsub(x = names(suppMat), pattern = "\\.", replacement = "+")
                                
                                
                                coefMat <- data.frame("mrg_L0" = d$coef_L0Mrg / d$coef_oseL0,
                                                      "mrg_rdg" = d$coef_mrgRdg / d$coef_oseL0,
                                                      "CS+Bbar" = d$coef_msP1_L0 / d$coef_oseL0,
                                                      "Zbar+L2" = d$coef_msP2_L0 / d$coef_oseL0,
                                                      "CS+L2" = d$coef_msP3_L0 / d$coef_oseL0,
                                                      "Bbar" = d$coef_msP4_L0 / d$coef_oseL0,
                                                      "Zbar+Bbar" = d$coef_msP5_L0 / d$coef_oseL0,
                                                      "SGL" = d$SGL_coef / d$coef_oseL0,
                                                      "GL" = d$GL_coef / d$coef_oseL0,
                                                      "grMCP" = d$grMCP_coef / d$coef_oseL0,
                                                      "gel" = d$gel_coef / d$coef_oseL0,
                                                      "cMCP" = d$cMCP_coef / d$coef_oseL0,
                                                      "Lasso" = d$lasso_MT / d$coef_oseL0)
                                
                                names(coefMat) <- gsub(x = names(coefMat), pattern = "\\.", replacement = "+")
                                
                                f1_mat <- data.frame("mrg_L0" = f1score(d, "mrgL0") / f1score(d, "oseL0"),
                                                      "CS+Bbar" = f1score(d, "msP1_L0") / f1score(d, "oseL0"),
                                                      "Zbar+L2" = f1score(d, "msP2_L0") / f1score(d, "oseL0"),
                                                      "CS+L2" = f1score(d, "msP3_L0") / f1score(d, "oseL0"),
                                                      "Bbar" = f1score(d, "msP4") / f1score(d, "oseL0"),
                                                      "Zbar+Bbar" = f1score(d, "msP5") / f1score(d, "oseL0"), 
                                                 "SGL" = f1score(d, "SGL") / f1score(d, "oseL0"),
                                                 "GL" = f1score(d, "GL") / f1score(d, "oseL0"),
                                                 "grMCP" = f1score(d, "grMCP") / f1score(d, "oseL0"),
                                                 "gel" = f1score(d, "gel") / f1score(d, "oseL0"),
                                                 "cMCP" = f1score(d, "cMCP") / f1score(d, "oseL0"),
                                                 "Lasso" = f1score(d, "lasso") / f1score(d, "oseL0")
                                                 )
                                
                                # make sure the names line up with suppMat
                                names(f1_mat) <- names(suppMat)[names(suppMat) != "Trace"] 
                                
                                tpMat <- data.frame(oseL0 = d$oseL0_tp,
                                                    ms1 = d$msP1_L0_tp,
                                                    ms2 = d$msP2_L0_tp,
                                                    ms3 = d$msP3_L0_tp,
                                                    "CS+Bbar" = d$msP1_L0_tp,
                                                    "Zbar+L2" = d$msP2_L0_tp,
                                                    "CS+L2" = d$msP3_L0_tp,
                                                    "Bbar" = d$msP4_tp,
                                                    "Zbar+Bbar" = d$msP5_tp,
                                                    "SGL" = d$SGL_tp,
                                                    "GL" = d$GL_tp,
                                                    "grMCP" = d$grMCP_tp,
                                                    "gel" = d$gel_tp,
                                                    "cMCP" = d$cMCP_tp,
                                                    "Lasso" = d$lasso_tp)
                                
                                names(tpMat) <- gsub(x = names(tpMat), pattern = "\\.", replacement = "+")
                                
                                
                                fpMat <- data.frame(mrgL0 = d$mrgL0_fp,
                                                    oseL0 = d$oseL0_fp,
                                                    ms1 = d$msP1_L0_fp,
                                                    ms2 = d$msP2_L0_fp,
                                                    ms3 = d$msP3_L0_fp,
                                                    "CS+Bbar" = d$msP1_L0_fp,
                                                    "Zbar+L2" = d$msP2_L0_fp,
                                                    "CS+L2" = d$msP3_L0_fp,
                                                    "Bbar" = d$msP4_fp,
                                                    "Zbar+Bbar" = d$msP5_fp,
                                                    "SGL" = d$SGL_fp,
                                                    "GL" = d$GL_fp,
                                                    "grMCP" = d$grMCP_fp,
                                                    "gel" = d$gel_fp,
                                                    "cMCP" = d$cMCP_fp,
                                                    "Lasso" = d$lasso_fp)
                                
                                names(fpMat) <- gsub(x = names(fpMat), pattern = "\\.", replacement = "+")
                                
                                sMat <- data.frame(mrgL0 = d$s_mrgL0,
                                                    oseL0 = d$s_ose,
                                                    "CS+Bbar" = d$s_MS3,
                                                    "Zbar+L2" = d$s_MS2,
                                                    "CS+L2" = d$s_MS1,
                                                    "Bbar" = d$s_MS4,
                                                    "Zbar+Bbar" = d$s_MS5,
                                                    "SGL" = d$SGL_s,
                                                    "GL" = d$GL_s,
                                                    "grMCP" = d$grMCP_s,
                                                    "gel" = d$gel_s,
                                                    "cMCP" = d$cMCP_s,
                                                    "Lasso" = d$lasso_s)
                                
                                names(sMat) <- gsub(x = names(sMat), pattern = "\\.", replacement = "+")
                                
                                
                                ls_beta[[cnt]] <- cbind( 
                                    gather(rmseMat), 
                                    b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
                                ls_beta2[[cnt]] <- cbind( 
                                  gather(rmseMat2), 
                                  b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
                                ls_supp[[cnt]] <- cbind( 
                                    gather(suppMat), 
                                    b,k,n, e1,r_exp,s,p,q, rho, mm )
                                
                                ls_fp[[cnt]] <- cbind( 
                                    gather(fpMat), 
                                    b,k,n, e1,r_exp,s,p,q, rho, mm  )
                                
                                ls_tp[[cnt]] <- cbind( 
                                    gather(tpMat), 
                                    b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
                                ls_coef[[cnt]] <- cbind( 
                                    gather(coefMat), 
                                    b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
                                ls_f1[[cnt]] <- cbind( 
                                    gather(f1_mat), 
                                    b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
                                ls_s[[cnt]] <- cbind( 
                                  gather(sMat), 
                                  b,k,n,e1,r_exp,s,p,q, rho, mm )
                                
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
}

####################################################################################################
# set colors
# group 1
levs <- grp_incl <- c("Bbar", "Lasso", "Zbar+L2", "gel", "grMCP", "Zbar+Bbar")
myColors <- setNames( c("#ca0020", "darkgrey", "#0868ac", "#525252", "#E69F00", "darkgreen"), levs) 

# group 2 - for appendix
levs2 <- grp_incl2 <- c("SGL", "CS+L2", "CS+Bbar", "Zbar+L2", "Zbar+Bbar", "cMCP")
myColors2 <- setNames( c("#ca0020", "darkgrey", "#0868ac", "#525252", "#E69F00", "darkgreen"), levs2) 

####################################################################################################
# factors
dat <- do.call(rbind, ls_beta)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$rho <- as.integer(dat$rho)
dat$r_exp <- as.factor(dat$r_exp)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)

###############################################
# rmse -- main text
###############################################
plt_r = dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( log(value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( log(value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +
  geom_hline(yintercept=0, 
             linetype="dashed", 
             color = "black", 
             size = rel(0.5),
             alpha = 0.7) + #
  ylab(TeX('${log(RMSE_{Method}/RMSE_{L_0 L_2})}$') )+ 
  xlab(TeX('Sparsity Level $(s)$')) + 
  scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
  scale_color_manual(values = myColors[levs %in% grp_incl] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5)) ) + 
  guides(fill= guide_legend(title="Method"))  

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( paste0("multiTaskRMSE_q0.pdf"),
        plot = plt_r,
        width = 12,
        height = 8)

plt_rmse <- plt_r

###############################################
# rmse -- main text
###############################################
cnt <- 1
plt_r = dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl2,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( log(value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( log(value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +
  geom_hline(yintercept=0, 
             linetype="dashed", 
             color = "black", 
             size = rel(0.5),
             alpha = 0.7) + #
  ylab(TeX('${log(RMSE_{Method}/RMSE_{L_0 L_2})}$') )+ 
  xlab(TeX('Sparsity Level $(s)$')) + 
  scale_fill_manual(values = myColors2[levs %in% grp_incl2] ) +
  scale_color_manual(values = myColors2[levs %in% grp_incl2] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5)) ) + 
  guides(fill= guide_legend(title="Method"))  

  setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
  ggsave( paste0("multiTaskRMSE_q0_supp.pdf"),
          plot = plt_r,
          width = 12,
          height = 8)
  
  assign(paste0("plt_r", cnt), plt_r)

###########################################
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

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")

#########################
# f1 -- main text
#########################
plt_f1 = dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( (value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( (value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +#,
  geom_hline(yintercept=1, 
             linetype="dashed", 
             color = "black", 
             size = rel(0.5),
             alpha = 0.7) + #
  ylab(TeX('${F1_{Method}/F1_{L_0 L_2}}$') )+ 
  xlab(TeX('Sparsity Level $(s)$')) + 
  scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
  scale_color_manual(values = myColors[levs %in% grp_incl] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5))
  ) + guides(fill= guide_legend(title="Method"))  
  
  
setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( paste0("multiTaskF1_q0.pdf"),
        plot = plt_f1,
        width = 12,
        height = 8)

f1_rmse <- ggarrange(plt_rmse, plt_f1, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( "multiTaskRMSE_F1_q0.pdf",
        plot = f1_rmse,
        width = 16,
        height = 8)


#########################
# fp and tp
#########################
# fp
dat <- do.call(rbind, ls_fp)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r_exp <- as.factor(dat$r_exp)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)
dat %>% tibble %>% 
  dplyr::group_by(key, b,k,n, e1) %>% 
  dplyr::summarize(my_mean = mean(value, na.rm = TRUE) ) %>% print(n = Inf)

plt_fp = 
  dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( (value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( (value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +
  ylab(TeX('False Positive Rate') )+ 
  xlab(TeX('Sparsity Level $(s)$')) + 
  scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
  scale_color_manual(values = myColors[levs %in% grp_incl] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5)) ) + 
  guides(fill= guide_legend(title="Method"))  

# tp
eRho <- 0.5
dat <- do.call(rbind, ls_tp)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$s <- as.factor(dat$s)
dat$r_exp <- as.factor(dat$r_exp)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)
dat %>% tibble %>% 
  dplyr::group_by(key, b,k,n, e1) %>% 
  dplyr::summarize(my_mean = mean(value, na.rm = TRUE) ) %>% print(n = Inf)

plt_tp = 
  dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( (value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( (value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +
  ylab(TeX('True Positive Rate') )+ 
  xlab(TeX('Sparsity Level $(s)$')) + 
  scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
  scale_color_manual(values = myColors[levs %in% grp_incl] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5)) ) + 
  guides(fill= guide_legend(title="Method"))  
f1_supp <- ggarrange(plt_fp, plt_tp, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( "multiTaskFP_TP_q0.pdf",
        plot = f1_supp,
        width = 16,
        height = 8)

#-------------------------------
# combined 
#-------------------------------
f1_supp <- ggarrange(plt_rmse, plt_f1, plt_fp, plt_tp, ncol=2, nrow=2, common.legend = TRUE, legend="bottom")

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( "multiTask_combined_q0.pdf",
        plot = f1_supp,
        width = 16,
        height = 20)






#---------------------------------------------
###########################################
dat <- do.call(rbind, ls_coef)
dat$b <- as.factor(dat$b)
dat$k <- as.factor(dat$k)
dat$n <- as.factor(dat$n)
dat$p <- as.factor(dat$p)
dat$e1 <- as.factor(2 * dat$e1)

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")

#########################
# coef - main text
#########################

plt_coef = dat %>% tibble %>%  
  dplyr::filter(q <= 20,
                e1 %in% c(1),
                n %in% c(50, 100),
                key %in% grp_incl,
                value != 0,
                k %in% c(4) ) %>%
  dplyr::group_by(b,k,n,e1,q, rho, r_exp, key) %>% 
  dplyr::summarize(my_mean = mean( (value), na.rm = TRUE),
                   sims = n(),
                   sd = sd( (value), na.rm = TRUE), 
                   se = sd/sqrt(sims) ) %>% 
  arrange(b, k,  rho, q, n, r_exp) %>%
  rename("Method"="key" ) %>%
  ggplot(aes( y = my_mean, x = rho, fill = Method, color = Method )) +
  facet_grid(n ~ r_exp, scales = "free_y") + 
  geom_line(size = 0.4) +
  geom_point(aes(shape = Method))+
  geom_errorbar(aes(ymin=my_mean-se, ymax=my_mean+se), linewidth = 0.35) +
  geom_hline(yintercept=1, 
             linetype="dashed", 
             color = "black", 
             size = rel(0.5),
             alpha = 0.7) + #
  ylab(TeX('${RMSE_{Method}/RMSE_{L_0 L_2}}$') )+ 
  xlab(TeX('$Support Level (s)$')) + 
  scale_fill_manual(values = myColors[levs %in% grp_incl] ) +
  scale_color_manual(values = myColors[levs %in% grp_incl] ) +
  scale_x_continuous(breaks = rhoVec) +
  theme_classic(base_size = 12) +
  coord_cartesian(ylim = c(0.15, 1) ) + 
  theme( plot.title = element_text(hjust = 0.5, color="black", size=rel(1.5), face="bold"),
         axis.text=element_text(face="bold",color="black", size=rel(1.5)),
         axis.title = element_text(face="bold", color="black", size=rel(1.5)),
         legend.key.size = unit(2, "line"), # added in to increase size
         legend.text = element_text(face="bold", color="black", size = rel(1.5)), # 3 GCL
         legend.title = element_text(face="bold", color="black", size = rel(1.5)),
         strip.text.x = element_text(face="bold", color="black", size = rel(1.5))
  ) + guides(fill= guide_legend(title="Method"))  

setwd("/Users/loewingergc/Desktop/NIMH Research/Sparse Multi-Study/Figures/Figure 6 - Sims support heterogeneity (q) vs performance/Resubmission/Final/Figures")
ggsave( paste0("multiTaskCoef_q0.pdf"),
        plot = plt_coef,
        width = 12,
        height = 8)
