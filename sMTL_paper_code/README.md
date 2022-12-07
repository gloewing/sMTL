# Sparse Multi-Task Learning Code

## Workflow
1) Open "~/Rcode/Intro Support Figure1 Demo.R" and run entire script.

Note that running this code requires specifying (in this .R file) the following path that is computer-specific:
juliaPath: the path name to Julia binary on your computer (line 16 of this .R file)
--To find the binary location, open Julia and enter the command: >> Base.julia_cmd()[1]
--Paste the pathname of that in the .R file

The folders are organized as follows:

### Figures 
1) Contains the .pdf files of Figures 1 and 3 in the main text. These figures are produced by the code in "~/Rcode/Intro Support Figure1 Demo.R"

### Rcode
1) "Intro Support Figure1 Demo.R" simulates data, runs sparse MTL functions (called from Julia) and plots the results in Figures 1 and 3 of the manuscript. 
2) "sparseFn_iht_test_MT.R" contains R code to tune MTL models
3) "betas" is a CSV procuded by "Intro Support Figure1 Demo.R" and can be used to produce the figures without running the MTL algorithms.

### Julia_code
1) Contains the Julia files called by R code.
