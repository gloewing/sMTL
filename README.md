# Multi-Task Learning for Sparsity Pattern Heterogeneity: A Discrete Optimiztion Approach (sMTL)
## Repository Description

Repository for the development version of the R Package `sMTL` as well as code to run analyses and reproduce figures in the associated manuscript.

## sMTL R Package

## Installation

The development version of the R Package `sMTL` can be downloaded as follows:

```{R}
library(devtools)
install_github("gloewing/sMTL", subdir = "Package/sMTL")
```

###  Package Usage
For a tutorial, please refer to [sMTL's Vignette](https://github.com/gloewing/sMTL/tree/main/Package/sMTL/vignettes). 

<br />

## Repository Folders
1) The 'Package' folder has code for the 'sMTL' R package that is current under development. This includes Julia code to run algorithms and R wrapper code to tune and fit these models. This package is still under development and has not been widely tested.

2) The 'sMTL_Paper' folder has code to run analyses and make figures for the manuscript. Some files will require changing path names. Please feel free to reach out to the author for more annotation or help with reproducing any analyses.

3) The 'sMTL_paper_Fig1_demo' folder contains code that can be used as a short self-contained demo/introduction to many of the methods proposed in the paper. This code should reproduce Figures 1 and 3 in the manuscript. It includes R code for data simulation, calls the algorithms (through Julia) to tune and fit models on the data and then generate figures.

<br />

## Dataset Links
Links to the data repositories for the neuroscience and cancer genomics applications can be found at:

1) Neuroscience application: https://osf.io/tb8fx/

Note that this data was originally from:

Gabriel Loewinger, Prasad Patil, Kenneth T Kishida, Giovanni Parmigiani. (2022)
Hierarchical resampling for bagging in multistudy prediction with applications to human neurochemical sensing. Annals of Applied Statistics. 16(4):2145-2165.

2) Breast Cancer application: https://osf.io/k6ynp/

Note that this data was a pre-processed version of data from:

PLANEY, K. (2020). curatedBreastData: Curated breast cancer gene expression data with survival and treatment information R package version 2.18.0.
