# Sparse Multi-Task Learning Code

## Figures 
1) Contain the .pdf files of Figures 1 and 3 int he main text. These files are produced by the code in "~/Rcode/Intro Support Figure1 Demo.R"

## Rcode
1) "sparseFn_iht_test_MT.R" contains R code to tune MTL models. 
2) "Intro Support Figure1 Demo.R" simulates data, runs sparse MTL functions (called from Julia) and plots the results in Figures 1 and 3. 
Running this code requires specifying in the R code the following that are specific to your computer:
juliaPath: the path name to Julia binary on your computer
--To find the binary location, open Julia and enter the command: "Base.julia_cmd()[1]"


## Julia_code
1) Contains the Julia files called by R code.

