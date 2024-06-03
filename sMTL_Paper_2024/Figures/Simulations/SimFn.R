
####*************************************************************************************
# "Simulations
####*************************************************************************************
package <- "mixtools"
if (!require(package, character.only=T, quietly=T)) {
    install.packages(package, 
                     repos='http://cran.us.r-project.org',
                     dependencies = TRUE
    )
    library(package, character.only=T)
}

mixGauss <- function(n, mu, sd, probs){
  
  K <- length(probs)
  vec <- vector(length = K, "list")
  mix <- sample.int(K, n, prob = probs, replace = TRUE) # miture probabilities
  for(j in 1:K)     vec[[j]] <- rnorm( n = sum(mix == j), mean = mu[j], sd = sd[j])
  
  return(unlist(vec))
}
###################################
# generate arbitrary \Sigma - NEW
###################################
# Fix the covariance matrix across studies and observations

multiStudySimNew <- function(sampSize = 50,
                          covariate.var = 1, # scales the variance of the MVNormal that generates the true means of the covaraites
                          beta.var = 1, # variance of random effects
                          cluster.beta.var = beta.var / 5, # variance of cluster specific random effect
                          cluster.X.var = covariate.var / 5, # variance of X specific random effect
                          clusters_mu = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                          clusters_beta = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                          num.covariates = 20,
                          zero_covs = c(), # indices of covariates which are 0
                          fixed.effects = c(), # indices of fixed effects
                          fixed.effectsX = c(), # indices of "fixed covariates" across studies
                          fixB = TRUE, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of betas marginally)
                          rfB = TRUE, # study specific boolean (if TRUE then there is study specific random effects for betas)
                          cB = TRUE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for betas)
                          fixX = TRUE, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of covariates marginally)
                          rfX = TRUE, # study specific boolean (if TRUE then there is study specific random effects for X)
                          cX = TRUE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for X)
                          studyNoise = c(1, 1), # range of noises for the different studies
                          num.studies = 5,
                          beta.mean.range = c(0.5, 5), # true means of hyperdistribution of beta are drawn from a unif(-beta.mean.range, beta.mean.range)
                          params = TRUE,
                          sigmaDiag = FALSE, # if true then the covariance matrix of covariates (X and Z) is diagonal
                          sigmaIdentity = FALSE, # if true then covariance matrix is the identity matrix
                          Xmeans_constant = FALSE, # if TRUE then the means of all covariates in a study are shifted by the same amount
                          XZmeans_constant = FALSE, # if TRUE then the means of all the X (fixed effects) covariates and Z (random effect) covariates each separately have all their means the same (all Xs same and all Zs same but X and Z different)
                          Xmean0 = FALSE, # IF TRUE then the marginal distribution of the Xs is mean 0
                          seedFixed = NA, # this seed is set right before drawing the fixed effects, if NA then no seed is set
                          seed = NA, # this is a general seed set at start (and after seedFixed)
                          covariance = "random", # "random"- qr decomposition, "identity", "exponential rho", "pariwise rho"
                          corr_rho = 0.5, # used if pariwise or exponential correlation
                          intercept = TRUE, # include intercept in model
                          uniform = FALSE # if unform, betas are drawn from uniform instead of Normal and variance is used as bounds
                          ){

    library(MASS) # for multivariate normal

    if(sigmaIdentity) covariance <- FALSE

    ###################
    # Covariance Matrix
    ###################
    if(!is.na(seedFixed))    set.seed(seedFixed) # set fixed effects seed if specified so fixed effects (and Sigma) are the same

    # fixed for all studies in this iteration
    ##################################
    # covariance matrix of covariates
    ##################################
    if(covariance == "random"){
        n <- num.covariates
        p <- qr.Q(qr(matrix(rnorm(n^2), n)))
        sig.vec <- abs(rnorm(n)) # diagonal elements so variances of the covariates
        Sigma <- crossprod(p, p*(sig.vec))
    }else if(covariance == "exponential"){
        Sigma <- matrix(1, ncol = num.covariates, nrow = num.covariates)

        # exponential correlation
        for(i in 1:num.covariates){
            for(j in 1:num.covariates){
                Sigma[i,j] <- corr_rho^( abs(i - j) )
            }
        }

    }else if(covariance == "pariwise"){
        Sigma <- matrix(corr_rho, ncol = num.covariates, nrow = num.covariates)
        diag(Sigma) <- 1
    }

    if( !covariance %in% c("pariwise", "exponential")    ){
        # only let Sigma be identity or diagonal matrix if not pariwise or exponential to avoid problems
        # because these are already identity on the diagonal so its meaningless
        if(sigmaDiag){
            # make random matrix and just take diagonals
            n <- num.covariates
            p <- qr.Q(qr(matrix(rnorm(n^2), n)))
            sig.vec <- abs(rnorm(n)) # diagonal elements so variances of the covariates
            Sigma <- crossprod(p, p*(sig.vec))
            Sigma <- diag( diag( Sigma )  ) # remove off diagonal terms
        }

    }

    if(sigmaIdentity)    Sigma <- diag( num.covariates ) # make identity matrix


    # set random seed unique to this iteration (assuming arguments vary across iteration)
    if(!is.na(seed))    set.seed(seed) # set random effects seed if specified so the rest is random

    if(length(fixed.effects) > 0){

        # length of fixed effects not including intercept
        if( is.element(0, fixed.effects) ){
            # do not count the intercept
            fixedEffLength <- length(fixed.effects) - 1
        }else{
            fixedEffLength <- length(fixed.effects)
        }

        # shift indices of fixed effects so now "1" is actually the intercept (which user indicates with 0)
        fixed.effects <- fixed.effects + 1 # if fixed.effects = 0, that is intercept so add 1 to all indices

    }else{
        fixedEffLength <- 0
    }

    ######################################
    # Variance of Random Effects and X random effects
    ######################################
    #### Varies between Studies
    if( length(beta.var) != num.covariates + 1){
        # if the length of the vector does not match, just use the first element
        message("Length of beta.var not equal to number of model coefficients: only the first element used for all coefficients")
        beta.var <- rep(beta.var[1], num.covariates + 1)
    }

    if( length(cluster.beta.var) != num.covariates + 1){
        # if the length of the vector does not match, just use the first element
        message("Length of cluster.beta.var not equal to number of model coefficients: only the first element used for all coefficients")
        cluster.beta.var <- rep(cluster.beta.var[1], num.covariates + 1)
    }

    if( length(cluster.X.var) != num.covariates){
        # if the length of the vector does not match, just use the first element
        message("Length of cluster.X.var not equal to number of model coefficients: only the first element used for all coefficients")
        cluster.X.var <- rep(cluster.X.var[1], num.covariates)
    }
    ###########################################################################
    # True means and variances of the covariates
    ###########################################################################
    # marginal means of covariates
    if(!Xmean0){
        mean.vec <- rnorm(num.covariates, 0, sqrt(10)) # the true means of the MVRN used to generate the Xs
    }else{
        mean.vec <- rep(0, num.covariates)
    }

    # marginal variance of covariates
    # keep degree of variability across studies constant across covariates
    mu.sigma.vec <- rep(1, num.covariates)

    ##############################
    # number of observations
    ##############################
    # number of observations per study
    obs.vec <- vector(length = num.studies)

    if( length(sampSize) == 1){
        obs.vec <- rep(sampSize, num.studies) # number of observations per study is constant
    }else if(length(sampSize) !=  num.studies){

        message("Sample Sizes provided not equal to total studies: Using first sample size provided for all studies")
        obs.vec <- rep(sampSize[1], num.studies) # number of observations per study is constant

    }else if (length(sampSize) ==  num.studies ){
        # if equal then just use that
        obs.vec <- sampSize

    }

    ##############################
    # model coefficients
    ##############################
    #***************
    # fixed effects
    #***************
    if(!is.na(seedFixed))    set.seed(seedFixed) # set fixed effects seed if specified so fixed effects are the same
    # row corresponds to study and column corresponds to True Beta for corresponding covaraite
    beta.matrix <- matrix(NA, nrow = num.studies, ncol = num.covariates + 1)

    # true betas (i.e., random effects are centered at these beta values)
    beta.mean.vec <- runif(num.covariates + 1, -beta.mean.range[2], beta.mean.range[2])

    #### bound away from 0 by finding those in range around 0 and draw another uniform
    # positive ones \in [0, bound]
    indx <- which(beta.mean.vec <= beta.mean.range[1] & beta.mean.vec >= 0)
    beta.mean.vec[indx] <- runif( length(indx),  beta.mean.range[1], beta.mean.range[2])

    # negative ones \in [-bound, 0]
    indx <- which(beta.mean.vec >= -beta.mean.range[1] & beta.mean.vec <= 0)
    beta.mean.vec[indx] <- -runif( length(indx),  beta.mean.range[1], beta.mean.range[2])

    # matrix of fixed effects
    beta.matrix <- t( replicate( num.studies, beta.mean.vec )   )

    #***************
    # random effects
    #***************
    # set seed again since we (maybe) set it for fixed effects above
    if(!is.na(seed))    set.seed(seed) # set fixed effects seed if specified so fixed effects are the same

    # different random effect for each study (default is 0)
    random.effects <- matrix(0, nrow = num.studies, ncol = num.covariates + 1)
    # assume indpendence between all random effects of all covariates of all studies
    # (cluster specific random effects below in next section)

    # draweit as a multivariate normal so we can use potentially different random effects
    if(!is.null(fixed.effects)){
        # if there are fixed effects
        random.effects[,-fixed.effects] <- mvrnorm(num.studies,
                                                   mu = rep(0, times = num.covariates + 1 - length(fixed.effects)  ),
                                                   Sigma = diag(beta.var[-fixed.effects])
        )
    }else{
        # if no fixed effects (same as above but without code that removes fixed effects)
        random.effects <- mvrnorm(num.studies,
                                   mu = rep(0, times = num.covariates + 1 ),
                                   Sigma = diag(beta.var)
                                )
    }


    #******************************
    # cluster random effects
    #******************************
    clustBeta <-  matrix(0, nrow = num.studies, ncol = num.covariates + 1) # different random effect for each cluster (default is 0)

    # if clusters then make matrix otherwise keep it as 0
    if( !is.null(clusters_beta) ){

        num.clusters <- length( unique( clusters_beta ) ) # numnber of unique clusters

        # only add these random effects if there are fewer clusters than studies
        if( num.clusters < num.studies ){

            # test whether there are fixed.effects, if there aren't then alter code so it does not throw errors (simualtion is still same)
            if( !is.null(fixed.effects) ){

                # draw cluster specific random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates + 1 - length(fixed.effects)  ),
                                    Sigma = diag(cluster.beta.var[-fixed.effects])
                                    )

                # iterate through and make each study have their cluster-specific random effect
                for (i in 1:num.studies){
                    clustBeta[i, -fixed.effects] <- clustMat[ clusters_beta[i],  ]  # clusters_beta[i] is the cluster number of the ith study
                }

                rm(clustMat)

            }else{
                # if no fixed.effects than remove code about it

                # draw cluster specific random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates + 1 ),
                                    Sigma = diag(cluster.beta.var)
                )

                # iterate through and make each study have their cluster-specific random effect
                for (i in 1:num.studies){
                    clustBeta[i, ] <- clustMat[ clusters_beta[i],  ]  # clusters_beta[i] is the cluster number of the ith study
                }

                rm(clustMat)
            }


        }

    }

    ########################
    # zero-out coefficients
    ########################
    if(is.matrix(zero_covs)){
        # if its a matrix it should be K x p  with 1s and 0s
        beta.matrix <- beta.matrix * zero_covs
        random.effects <- random.effects * zero_covs
        clustBeta <- clustBeta * zero_covs

    }else if(is.vector(zero_covs)){
        if(length(zero_covs) > 0 ){
            # if its a vector its an index for each covaraite (should be p x 1)
            beta.matrix[,zero_covs] <- random.effects[,zero_covs] <- clustBeta[,zero_covs] <- 0
        }
    }

    #-------------------------------------------------------------------------
    # full coefficients (fixed + random effects + cluster-specific random effects)
    # multiplied by booleans (fixB, rfB, cB) which zero out the corresponding components
    betas <- beta.matrix * fixB + random.effects * rfB + clustBeta * cB
    #-------------------------------------------------------------------------

    ##########################
    # Generate Covariates
    ##########################
    # Rows: Study Number
    # Columns: Covariate
    #*****************
    # fixed effects X
    #*****************
    if( length(covariate.var) != num.covariates){
        # if the length of the vector does not match, just use the first element
        message("Length of covariate.var not equal to number of covariates: only the first element used for all coefficients")
        covariate.var <- rep(covariate.var[1], num.covariates)
    }

    # row corresponds to study and column corresponds to True Beta for corresponding covaraite
    # generate true covariate means across studies within covariate
    mu.matrix <- matrix(NA, nrow = num.studies, ncol = num.covariates)
    #
    # # true means of Xs (i.e., Xs are centered at these values)

    # matrix of X "fixed effects"
    mu.matrix <- t( replicate( num.studies, mean.vec)   )

    #******************
    # random effects X
    #******************
    # different random effect for each study (default is 0 which corresponds to no "random effect")
    random.effectsX <- matrix( 0, nrow = num.studies, ncol = num.covariates )

    # assume indpendence between all random effects of all covariates of all studies

    # drawit as a multivariate normal so we can use potentially different random effects variances
    if(!is.null(fixed.effectsX)){
        # if there are "fixed effects" for X
        random.effectsX[, -fixed.effectsX] <- mvrnorm(num.studies,
                                                mu = rep(0, times = num.covariates - length(fixed.effectsX)  ),
                                                Sigma = diag( covariate.var[-fixed.effectsX] )
                                                )
    }else{
        # if no fixed effects -- same but without code that removes fixedEffectsX
        random.effectsX <- mvrnorm(num.studies,
                                   mu = rep(0, times = num.covariates ),
                                   Sigma = diag( covariate.var )
                                    )
    }

    #******************************
    # cluster random effects X
    #******************************
    #### Varies between Studies

    clustX <-  matrix(0, nrow = num.studies, ncol = num.covariates )

    # different random effect for each cluster (default is 0)
    # only add these random effects if there are fewer clusters than studies
    if( !is.null(clusters_mu) ){

        num.clusters <- length( unique( clusters_mu ) )

        if(num.clusters < num.studies ){

            # if clusters then make matrix
            num.clusters <- length( unique( clusters_mu ) ) # number of unique clusters

            # iterate through and make each study have their cluster-specific random effect
            if(length(fixed.effectsX) > 0 ){
                # if there are fixed effects

                # draw cluster specific X random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates - length(fixed.effectsX)  ),
                                    Sigma = diag( cluster.X.var[-fixed.effectsX] )
                )

                for (i in 1:num.studies){
                    clustX[i, -fixed.effectsX] <- clustMat[ clusters_mu[i],  ]  # clusters_mu[i] is the cluster number of the ith study
                }

                rm(clustMat)

            }else{
                # if no fixed effects (take out code that removes)


                # draw cluster specific X random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates  ),
                                    Sigma = diag( cluster.X.var )
                )
                for (i in 1:num.studies){
                    clustX[i,] <- clustMat[ clusters_mu[i],  ]  # clusters_mu[i] is the cluster number of the ith study
                }

                rm(clustMat)
            }

        }
    }

    #######################
    # simulate X in simple case where means of covariates are shifted the same for each covariate in a study
    #######################
    # this means that the mean of all covariates in a study are the same
    if( XZmeans_constant | Xmeans_constant ){

        # shifts means of X if length of fixed effects (not including intercept) is > 0
        if( fixedEffLength > 0 ){

            if(is.element(1, fixed.effects)){
                # check to see if intercept is a fixed effect
                fixedEffectsIndx <- fixed.effects[-1] - 1 # removes intercept and uses covariate indices
            }else{
                fixedEffectsIndx <- fixed.effects - 1 # uses covariate indices
            }

            randXIndx <- seq(1, ncol(mu.matrix) )[-fixedEffectsIndx] # indices of covariates of Z (design matrix for random effects)

            # select the first covariate's means of each study to be the same for all covariates
            # do this for each of the types (randomEffectsX, fixedEffectsX) so the means are the same for each covaraite

            # arbitrarily choose the first covariate's means to be the means of all the covariates in the study:
            # ** fixedEffectsX
            mu.matrix[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                    mu.matrix[,1])
            # ** randomEffectsX
            random.effectsX[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                          random.effectsX[,1])

            # ** cluster specific randomEffectsX
            clustX[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                 clustX[,1])
        }else{
            # if there are no fixed effects
            fixedEffectsIndx <- c()
            randXIndx <- seq(1, ncol(mu.matrix) )

            # # ** randomEffectsX
        }

        # shifts means of Z (if XZmeans_constant = TRUE)
        if( length(fixedEffectsIndx) < num.covariates &
            XZmeans_constant
            ){
            # make sure there are still random effects (i.e., there exists a Z matrix)

            # select the first covariate's means of each study to be the same for all covariates
            # do this for each of the types (randomEffectsX, fixedEffectsX) so the means are the same for each covaraite

            # arbitrarily choose the first covariate's means of Z (design matrix for random effects) to be the means of all the covariates in the study -- randXIndx[1]
            if(length(fixedEffectsIndx) > 0){ # make sure there are fixed effects)
            # ** fixedEffectsX
                mu.matrix[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                        mu.matrix[, randXIndx[1] ])
                # ** randomEffectsX
                random.effectsX[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                              random.effectsX[, randXIndx[1] ])

                # ** cluster specific randomEffectsX
                clustX[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                     clustX[, randXIndx[1] ])
            }else{
                # if there are no fixed effects

                # ** fixedEffectsX
                mu.matrix <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                            mu.matrix[, randXIndx[1] ])
                # ** randomEffectsX
                random.effectsX <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                                  random.effectsX[, randXIndx[1] ])

                # ** cluster specific randomEffectsX
                clustX <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                         clustX[, randXIndx[1] ])
            }
        }

    }


    #-------------------------------------------------------------------------
    # full matrix of means of coefficients (fixX, rfX and cX are booleans which 0 out those matrices if desirable)
    muX <- mu.matrix * fixX + random.effectsX * rfX + clustX * cX
    #-------------------------------------------------------------------------

    if( XZmeans_constant | Xmeans_constant ){

        d <- muX[,1]
        if(length(fixedEffectsIndx) > 0){
            # if there are fixed effects
            randXIndx <- seq(1, ncol(muX) )[-fixedEffectsIndx] # indices of covariates of Z (design matrix for random effects)

        }else{
            randXIndx <- seq(1, ncol(muX) ) # indices of covariates of Z (design matrix for random effects)
        }

        if( XZmeans_constant)      d <- cbind(d, muX[, randXIndx[1] ]  ) # covariates of random effect shift in covariate mean -- use fixed.effects[1] since we arbitrarily chose the first one above
    }else{
        d <- NA
    }
#########################################################################

    #########################################################
    # Variance of error terms for different studies
    #########################################################
    # different variance levels of \epsilon for different studies
    noiseVec <- vector(length = num.studies)
    for(j in 1:num.studies){

        noiseVec[j] <- runif(1, studyNoise[1], studyNoise[2])

    }

    noiseVec <- sqrt( noiseVec ) # make it so it is on standard deviation scale to sample from rnorm() below

    #########################################################
    # simulate design matrix and outcome conditional on parameters above
    #########################################################

    # Simulate data with above parameters
    studies <- vector("list", length = num.studies) # each element is a study

    for(y in 1:num.studies){

        # data.sim <- matrix(NA, ncol = num.covariates, nrow = obs.vec[y])

        ##############################
        # Generate the design matrix
        ##############################
        data.sim <- mvrnorm(obs.vec[y], muX[y , ], Sigma)

        # generate a vector Y and add noise
        if(intercept){
            Y <- cbind(1, data.sim) %*% betas[y, ] + rnorm(obs.vec[y], 0, noiseVec[y]) # noise is mean 0 with study specific noise levels
        }else{
            # if no intercept
            Y <- data.sim %*% betas[y, -1] + rnorm(obs.vec[y], 0, noiseVec[y]) # noise is mean 0 with study specific noise levels
        }
        # bind it to data
        studies[[y]] <- cbind(y, Y, data.sim) # first column is study number, then Y outcome, then design matroix
        colnames(studies[[y]]) <- c("Study", "Y", paste0("V_", 1:num.covariates))

    }

    rm(data.sim, Y)

    # if no intercept remove terms corresponding to it
    if(!intercept){
        betas = betas[,-1]
        beta.matrix = beta.matrix[,-1]
        random.effects = random.effects[,-1]
        clustBeta = clustBeta[,-1]
        beta.mean.vec <- beta.mean.vec[-1]
    }



    # concatenate ("merge") all studies together
    final.results <- do.call(rbind, studies)

    # colnames
    colnames(final.results) <- c("Study", "Y", paste0("V_", 1:num.covariates))

    message(paste0("Data Simulation Complete"))

    if(params){
        return( list(data = final.results,
                     betaMeans = beta.mean.vec,
                     betas = betas,
                     fixedBetas = beta.matrix,
                     randEff = random.effects,
                     clustRandEff = clustBeta,
                     xMean = mean.vec,
                     Xs = muX,
                     fixedX = mu.matrix,
                     randX = random.effectsX,
                     clustRandEffX = clustX,
                     Sigma = Sigma,
                     d = d))
    }else{
        return( final.results )
    }




}


###################################
# generate arbitrary \Sigma - NEW
###################################
# Fix the covariance matrix across studies and observations
# multi task version with same design matrix

multiStudySimNew_MT <- function(sampSize = 50,
                             covariate.var = 1, # scales the variance of the MVNormal that generates the true means of the covaraites
                             beta.var = 1, # variance of random effects
                             cluster.beta.var = beta.var / 5, # variance of cluster specific random effect
                             cluster.X.var = covariate.var / 5, # variance of X specific random effect
                             clusters_mu = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                             clusters_beta = NULL, # vector of cluster where each element indicates which cluster that study is in (i.e., study is index of vector)
                             num.covariates = 20,
                             zero_covs = c(), # indices of covariates which are 0
                             fixed.effects = c(), # indices of fixed effects
                             fixed.effectsX = c(), # indices of "fixed covariates" across studies
                             fixB = TRUE, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of betas marginally)
                             rfB = TRUE, # study specific boolean (if TRUE then there is study specific random effects for betas)
                             cB = TRUE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for betas)
                             fixX = TRUE, # boolean of whether there is "fixed effects" (this basically means whether there is random variation in means of covariates marginally)
                             rfX = TRUE, # study specific boolean (if TRUE then there is study specific random effects for X)
                             cX = TRUE, # cluster specific boolean for random effects of X (if TRUE then there is cluster specific random effects for X)
                             studyNoise = c(1, 1), # range of noises for the different studies
                             num.studies = 5,
                             beta.mean.range = c(0.5, 5), # true means of hyperdistribution of beta are drawn from a unif(-beta.mean.range, beta.mean.range)
                             params = TRUE,
                             sigmaDiag = FALSE, # if true then the covariance matrix of covariates (X and Z) is diagonal
                             sigmaIdentity = FALSE, # if true then covariance matrix is the identity matrix
                             Xmeans_constant = FALSE, # if TRUE then the means of all covariates in a study are shifted by the same amount
                             XZmeans_constant = FALSE, # if TRUE then the means of all the X (fixed effects) covariates and Z (random effect) covariates each separately have all their means the same (all Xs same and all Zs same but X and Z different)
                             Xmean0 = FALSE, # IF TRUE then the marginal distribution of the Xs is mean 0
                             seedFixed = NA, # this seed is set right before drawing the fixed effects, if NA then no seed is set
                             seed = NA, # this is a general seed set at start (and after seedFixed)
                             covariance = "random", # "random"- qr decomposition, "identity", "exponential rho", "pariwise rho"
                             corr_rho = 0.5, # used if pariwise or exponential correlation
                             intercept = TRUE, # include intercept in model
                             mix_probs = NA, # mixture probabilities: if NA then no mixture of gaussians
                             mix_mu = NA # mixture means of mix of gaussians: if NA then no mixture of gaussians
){
    
    library(MASS) # for multivariate normal
    
    if(sigmaIdentity) covariance <- FALSE

    # Covariance Matrix
    ###################
    if(!is.na(seedFixed))    set.seed(seedFixed) # set fixed effects seed if specified so fixed effects (and Sigma) are the same
    
    # fixed for all studies in this iteration
    ##################################
    # covariance matrix of covariates
    ##################################
    if(covariance == "random"){
        n <- num.covariates
        p <- qr.Q(qr(matrix(rnorm(n^2), n)))
        sig.vec <- abs(rnorm(n)) # diagonal elements so variances of the covariates
        Sigma <- crossprod(p, p*(sig.vec))
    }else if(covariance == "exponential"){
        Sigma <- matrix(1, ncol = num.covariates, nrow = num.covariates)
        
        # exponential correlation
        for(i in 1:num.covariates){
            for(j in 1:num.covariates){
                Sigma[i,j] <- corr_rho^( abs(i - j) )
            }
        }
        
    }else if(covariance == "pariwise"){
        Sigma <- matrix(corr_rho, ncol = num.covariates, nrow = num.covariates)
        diag(Sigma) <- 1
    }
    
    if( !covariance %in% c("pariwise", "exponential")    ){
        # only let Sigma be identity or diagonal matrix if not pariwise or exponential to avoid problems
        # because these are already identity on the diagonal so its meaningless
        if(sigmaDiag){
            # make random matrix and just take diagonals
            n <- num.covariates
            p <- qr.Q(qr(matrix(rnorm(n^2), n)))
            sig.vec <- abs(rnorm(n)) # diagonal elements so variances of the covariates
            Sigma <- crossprod(p, p*(sig.vec))
            Sigma <- diag( diag( Sigma )  ) # remove off diagonal terms
        }
        
    }
    
    if(sigmaIdentity)    Sigma <- diag( num.covariates ) # make identity matrix
    
    
    # set random seed unique to this iteration (assuming arguments vary across iteration)
    if(!is.na(seed))    set.seed(seed) # set random effects seed if specified so the rest is random
    
    if(length(fixed.effects) > 0){
        
        # length of fixed effects not including intercept
        if( is.element(0, fixed.effects) ){
            # do not count the intercept
            fixedEffLength <- length(fixed.effects) - 1
        }else{
            fixedEffLength <- length(fixed.effects)
        }
        
        # shift indices of fixed effects so now "1" is actually the intercept (which user indicates with 0)
        fixed.effects <- fixed.effects + 1 # if fixed.effects = 0, that is intercept so add 1 to all indices
        
    }else{
        fixedEffLength <- 0
    }
    
    ######################################
    # Variance of Random Effects and X random effects
    ######################################
    #### Varies between Studies
    if( length(beta.var) != num.covariates + 1){
        # if the length of the vector does not match, just use the first element
        message("Length of beta.var not equal to number of model coefficients: only the first element used for all coefficients")
        beta.var <- rep(beta.var[1], num.covariates + 1)
    }
    
    if( length(cluster.beta.var) != num.covariates + 1){
        # if the length of the vector does not match, just use the first element
        message("Length of cluster.beta.var not equal to number of model coefficients: only the first element used for all coefficients")
        cluster.beta.var <- rep(cluster.beta.var[1], num.covariates + 1)
    }
    
    if( length(cluster.X.var) != num.covariates){
        # if the length of the vector does not match, just use the first element
        message("Length of cluster.X.var not equal to number of model coefficients: only the first element used for all coefficients")
        cluster.X.var <- rep(cluster.X.var[1], num.covariates)
    }
    ###########################################################################
    # True means and variances of the covariates
    ###########################################################################
    # marginal means of covariates
    if(!Xmean0){
        mean.vec <- rnorm(num.covariates, 0, sqrt(10)) # the true means of the MVRN used to generate the Xs
    }else{
        mean.vec <- rep(0, num.covariates)
    }
    
    # marginal variance of covariates
    # keep degree of variability across studies constant across covariates
    mu.sigma.vec <- rep(1, num.covariates)
    
    ##############################
    # number of observations
    ##############################
    # number of observations per study
    obs.vec <- vector(length = num.studies)
    
    if( length(sampSize) == 1){
        obs.vec <- rep(sampSize, num.studies) # number of observations per study is constant
    }else if(length(sampSize) !=  num.studies){
        
        message("Sample Sizes provided not equal to total studies: Using first sample size provided for all studies")
        obs.vec <- rep(sampSize[1], num.studies) # number of observations per study is constant
        
    }else if (length(sampSize) ==  num.studies ){
        # if equal then just use that
        obs.vec <- sampSize
        
    }
    
    ##############################
    # model coefficients
    ##############################
    #***************
    # fixed effects
    #***************
    if(!is.na(seedFixed))    set.seed(seedFixed) # set fixed effects seed if specified so fixed effects are the same
    # row corresponds to study and column corresponds to True Beta for corresponding covaraite
    beta.matrix <- matrix(NA, nrow = num.studies, ncol = num.covariates + 1)
    
    # true betas (i.e., random effects are centered at these beta values)
    beta.mean.vec <- runif(num.covariates + 1, -beta.mean.range[2], beta.mean.range[2])
    
    #### bound away from 0 by finding those in range around 0 and draw another uniform
    # positive ones \in [0, bound]
    indx <- which(beta.mean.vec <= beta.mean.range[1] & beta.mean.vec >= 0)
    beta.mean.vec[indx] <- runif( length(indx),  beta.mean.range[1], beta.mean.range[2])
    
    # negative ones \in [-bound, 0]
    indx <- which(beta.mean.vec >= -beta.mean.range[1] & beta.mean.vec <= 0)
    beta.mean.vec[indx] <- -runif( length(indx),  beta.mean.range[1], beta.mean.range[2])
    
    # matrix of fixed effects
    beta.matrix <- t( replicate( num.studies, beta.mean.vec )   )
    
    #***************
    # random effects
    #***************
    # set seed again since we (maybe) set it for fixed effects above
    if(!is.na(seed))    set.seed(seed) # set fixed effects seed if specified so fixed effects are the same
    
    # different random effect for each study (default is 0)
    random.effects <- matrix(0, nrow = num.studies, ncol = num.covariates + 1)
    # assume indpendence between all random effects of all covariates of all studies
    # (cluster specific random effects below in next section)

    # draw it as a multivariate normal so we can use potentially different random effects
    
    if(any(is.na(mix_probs))){
      message("Gaussian RE")
      # gaussian random effects
      
      if(!is.null(fixed.effects)){
        # if there are fixed effects
        random.effects[,-fixed.effects] <- mvrnorm(num.studies,
                                                   mu = rep(0, times = num.covariates + 1 - length(fixed.effects)  ),
                                                   Sigma = diag(beta.var[-fixed.effects]) )
      }else{
        # if no fixed effects (same as above but without code that removes fixed effects)
        random.effects <- mvrnorm(num.studies,
                                  mu = rep(0, times = num.covariates + 1 ),
                                  Sigma = diag(beta.var))
      }
    }else{
      # mixture of gaussians
      message("Mixture of Gaussians RE")
      
      if(!is.null(fixed.effects)){
        # if there are fixed effects
        for(jj in seq(1, num.covariates + 1)[-fixed.effects] ){
          random.effects[,jj] <- mixGauss(n = num.studies, 
                                          mu = mix_mu, 
                                          sd = rep(beta.var[jj], length(mix_probs)), # takes mixtures to have the same variance
                                          probs = mix_probs)
        }
      }else{
        # if no fixed effects (same as above but without code that removes fixed effects)
        for(jj in seq(1, num.covariates + 1)){
          random.effects[,jj] <- mixGauss(n = num.studies, 
                                          mu = mix_mu, 
                                          sd = rep(beta.var[jj], length(mix_probs)), # takes mixtures to have the same variance
                                          probs = mix_probs)
        }
      
      }
    }
    
    
    
    #******************************
    # cluster random effects
    #******************************
    clustBeta <-  matrix(0, nrow = num.studies, ncol = num.covariates + 1) # different random effect for each cluster (default is 0)
    
    # if clusters then make matrix otherwise keep it as 0
    if( !is.null(clusters_beta) ){
        
        num.clusters <- length( unique( clusters_beta ) ) # numnber of unique clusters
        
        # only add these random effects if there are fewer clusters than studies
        if( num.clusters < num.studies ){
            
            # test whether there are fixed.effects, if there aren't then alter code so it does not throw errors (simualtion is still same)
            if( !is.null(fixed.effects) ){
                
                # draw cluster specific random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates + 1 - length(fixed.effects)  ),
                                    Sigma = diag(cluster.beta.var[-fixed.effects])
                )
                
                # iterate through and make each study have their cluster-specific random effect
                for (i in 1:num.studies){
                    clustBeta[i, -fixed.effects] <- clustMat[ clusters_beta[i],  ]  # clusters_beta[i] is the cluster number of the ith study
                }
                
                rm(clustMat)
                
            }else{
                # if no fixed.effects than remove code about it
                
                # draw cluster specific random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates + 1 ),
                                    Sigma = diag(cluster.beta.var)
                )
                
                # iterate through and make each study have their cluster-specific random effect
                for (i in 1:num.studies){
                    clustBeta[i, ] <- clustMat[ clusters_beta[i],  ]  # clusters_beta[i] is the cluster number of the ith study
                }
                
                rm(clustMat)
            }
            
            
        }
        
    }
    
    ########################
    # zero-out coefficients
    ########################
    if(is.matrix(zero_covs)){
        # if its a matrix it should be K x p  with 1s and 0s
        beta.matrix <- beta.matrix * zero_covs
        random.effects <- random.effects * zero_covs
        clustBeta <- clustBeta * zero_covs
        
    }else if(is.vector(zero_covs)){
        if(length(zero_covs) > 0 ){
            # if its a vector its an index for each covaraite (should be p x 1)
            beta.matrix[,zero_covs] <- random.effects[,zero_covs] <- clustBeta[,zero_covs] <- 0
        }
    }
    
    #-------------------------------------------------------------------------
    # full coefficients (fixed + random effects + cluster-specific random effects)
    # multiplied by booleans (fixB, rfB, cB) which zero out the corresponding components
    betas <- beta.matrix * fixB + random.effects * rfB + clustBeta * cB
    #-------------------------------------------------------------------------
    
    ##########################
    # Generate Covariates
    ##########################
    # Rows: Study Number
    # Columns: Covariate
    #*****************
    # fixed effects X
    #*****************
    if( length(covariate.var) != num.covariates){
        # if the length of the vector does not match, just use the first element
        message("Length of covariate.var not equal to number of covariates: only the first element used for all coefficients")
        covariate.var <- rep(covariate.var[1], num.covariates)
    }
    
    # row corresponds to study and column corresponds to True Beta for corresponding covaraite
    # generate true covariate means across studies within covariate
    mu.matrix <- matrix(NA, nrow = num.studies, ncol = num.covariates)
    #
    # # true means of Xs (i.e., Xs are centered at these values)

    # matrix of X "fixed effects"
    mu.matrix <- t( replicate( num.studies, mean.vec)   )
    
    #******************
    # random effects X
    #******************
    # different random effect for each study (default is 0 which corresponds to no "random effect")
    random.effectsX <- matrix( 0, nrow = num.studies, ncol = num.covariates )
    
    # assume indpendence between all random effects of all covariates of all studies
    
    # drawit as a multivariate normal so we can use potentially different random effects variances
    if(!is.null(fixed.effectsX)){
        # if there are "fixed effects" for X
        random.effectsX[, -fixed.effectsX] <- mvrnorm(num.studies,
                                                      mu = rep(0, times = num.covariates - length(fixed.effectsX)  ),
                                                      Sigma = diag( covariate.var[-fixed.effectsX] )
        )
    }else{
        # if no fixed effects -- same but without code that removes fixedEffectsX
        random.effectsX <- mvrnorm(num.studies,
                                   mu = rep(0, times = num.covariates ),
                                   Sigma = diag( covariate.var )
        )
    }
    
    #******************************
    # cluster random effects X
    #******************************
    #### Varies between Studies
    
    clustX <-  matrix(0, nrow = num.studies, ncol = num.covariates )
    
    # different random effect for each cluster (default is 0)
    # only add these random effects if there are fewer clusters than studies
    if( !is.null(clusters_mu) ){
        num.clusters <- length( unique( clusters_mu ) )
        
        if(num.clusters < num.studies ){
            
            # if clusters then make matrix
            num.clusters <- length( unique( clusters_mu ) ) # number of unique clusters
            
            # iterate through and make each study have their cluster-specific random effect
            if(length(fixed.effectsX) > 0 ){
                # if there are fixed effects
                
                # draw cluster specific X random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates - length(fixed.effectsX)  ),
                                    Sigma = diag( cluster.X.var[-fixed.effectsX] )
                )
                
                for (i in 1:num.studies){
                    clustX[i, -fixed.effectsX] <- clustMat[ clusters_mu[i],  ]  # clusters_mu[i] is the cluster number of the ith study
                }
                
                rm(clustMat)
                
            }else{
                # if no fixed effects (take out code that removes)
                
                
                # draw cluster specific X random effects
                clustMat <- mvrnorm(num.clusters,
                                    mu = rep(0, times = num.covariates  ),
                                    Sigma = diag( cluster.X.var )
                )
                for (i in 1:num.studies){
                    clustX[i,] <- clustMat[ clusters_mu[i],  ]  # clusters_mu[i] is the cluster number of the ith study
                }
                
                rm(clustMat)
            }
            
        }
    }
    
    #######################
    # simulate X in simple case where means of covariates are shifted the same for each covariate in a study
    #######################
    # this means that the mean of all covariates in a study are the same
    if( XZmeans_constant | Xmeans_constant ){
        
        # shifts means of X if length of fixed effects (not including intercept) is > 0
        if( fixedEffLength > 0 ){
            
            if(is.element(1, fixed.effects)){
                # check to see if intercept is a fixed effect
                fixedEffectsIndx <- fixed.effects[-1] - 1 # removes intercept and uses covariate indices
            }else{
                fixedEffectsIndx <- fixed.effects - 1 # uses covariate indices
            }
            
            randXIndx <- seq(1, ncol(mu.matrix) )[-fixedEffectsIndx] # indices of covariates of Z (design matrix for random effects)
            
            # select the first covariate's means of each study to be the same for all covariates
            # do this for each of the types (randomEffectsX, fixedEffectsX) so the means are the same for each covaraite
            
            # arbitrarily choose the first covariate's means to be the means of all the covariates in the study:
            # ** fixedEffectsX
            mu.matrix[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                       mu.matrix[,1])
            # ** randomEffectsX
            random.effectsX[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                             random.effectsX[,1])
            
            # ** cluster specific randomEffectsX
            clustX[,fixedEffectsIndx] <- replicate( n = length(fixedEffectsIndx),
                                                    clustX[,1])
        }else{
            # if there are no fixed effects
            fixedEffectsIndx <- c()
            randXIndx <- seq(1, ncol(mu.matrix) )
            
            # # ** randomEffectsX

        }
        
        # shifts means of Z (if XZmeans_constant = TRUE)
        if( length(fixedEffectsIndx) < num.covariates &
            XZmeans_constant
        ){
            # make sure there are still random effects (i.e., there exists a Z matrix)
            
            # select the first covariate's means of each study to be the same for all covariates
            # do this for each of the types (randomEffectsX, fixedEffectsX) so the means are the same for each covaraite
            
            # arbitrarily choose the first covariate's means of Z (design matrix for random effects) to be the means of all the covariates in the study -- randXIndx[1]
            if(length(fixedEffectsIndx) > 0){ # make sure there are fixed effects)
                # ** fixedEffectsX
                mu.matrix[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                            mu.matrix[, randXIndx[1] ])
                # ** randomEffectsX
                random.effectsX[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                                  random.effectsX[, randXIndx[1] ])
                
                # ** cluster specific randomEffectsX
                clustX[,-fixedEffectsIndx] <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                                         clustX[, randXIndx[1] ])
            }else{
                # if there are no fixed effects
                
                # ** fixedEffectsX
                mu.matrix <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                        mu.matrix[, randXIndx[1] ])
                # ** randomEffectsX
                random.effectsX <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                              random.effectsX[, randXIndx[1] ])
                
                # ** cluster specific randomEffectsX
                clustX <- replicate( n = num.covariates - length(fixedEffectsIndx),
                                     clustX[, randXIndx[1] ])
            }
        }
        
    }
    
    
    #-------------------------------------------------------------------------
    # full matrix of means of coefficients (fixX, rfX and cX are booleans which 0 out those matrices if desirable)
    muX <- mu.matrix * fixX + random.effectsX * rfX + clustX * cX
    #-------------------------------------------------------------------------
    
    if( XZmeans_constant | Xmeans_constant ){
        
        d <- muX[,1]
        if(length(fixedEffectsIndx) > 0){
            # if there are fixed effects
            randXIndx <- seq(1, ncol(muX) )[-fixedEffectsIndx] # indices of covariates of Z (design matrix for random effects)
            
        }else{
            randXIndx <- seq(1, ncol(muX) ) # indices of covariates of Z (design matrix for random effects)
        }
        
        if( XZmeans_constant)      d <- cbind(d, muX[, randXIndx[1] ]  ) # covariates of random effect shift in covariate mean -- use fixed.effects[1] since we arbitrarily chose the first one above
    }else{
        d <- NA
    }
    #########################################################################
    
    #########################################################
    # Variance of error terms for different studies
    #########################################################
    # different variance levels of \epsilon for different studies
    noiseVec <- vector(length = num.studies)
    for(j in 1:num.studies){
        
        noiseVec[j] <- runif(1, studyNoise[1], studyNoise[2])
        
    }
    
    noiseVec <- sqrt( noiseVec ) # make it so it is on standard deviation scale to sample from rnorm() below
    
    #########################################################
    # simulate design matrix and outcome conditional on parameters above
    #########################################################
    
    # Simulate data with above parameters
    studies <- vector("list", length = num.studies) # each element is a study
    
    ##############################
    # Generate the design matrix -- foxed between studies
    ##############################
    data.sim <- mvrnorm(obs.vec[1], muX[1 , ], Sigma) # arbitrarily choose first "study" for number of observations and means of covariates
    colnames(data.sim) <- paste0( "V_", 1:ncol(data.sim) )
    Ymat <- matrix(NA, nrow = obs.vec[1], ncol = num.studies) # stores all the Ys
    colnames(Ymat) <- paste0("Y_", 1:num.studies) # each "study is a task
    
    for(y in 1:num.studies){
        
        # generate a vector Y and add noise
        if(intercept){
            Y <- cbind(1, data.sim) %*% betas[y, ] + rnorm(obs.vec[1], 0, noiseVec[y]) # noise is mean 0 with study specific noise levels
        }else{
            # if no intercept
            Y <- data.sim %*% betas[y, -1] + rnorm(obs.vec[1], 0, noiseVec[y]) # noise is mean 0 with study specific noise levels
        }
        # bind it to data
        Ymat[,y] <- Y # save outcom e

        
    }
    
    # concatenate ("merge") all studies together
    final.results <- cbind(Ymat, data.sim) # merge outcome matrix and design matrix 
    
    rm(data.sim, Y)
    
    # if no intercept remove terms corresponding to it
    if(!intercept){
        betas = betas[,-1]
        beta.matrix = beta.matrix[,-1]
        random.effects = random.effects[,-1]
        clustBeta = clustBeta[,-1]
        beta.mean.vec <- beta.mean.vec[-1]
    }
    
    # colnames
    colnames(final.results) <- c(paste0("Y_", 1:num.studies), paste0("V_", 1:num.covariates))  #c("Study", "Y", paste0("V_", 1:num.covariates))
    
    message(paste0("Data Simulation Complete"))
    
    if(params){
        return( list(data = final.results,
                     betaMeans = beta.mean.vec,
                     betas = betas,
                     fixedBetas = beta.matrix,
                     randEff = random.effects,
                     clustRandEff = clustBeta,
                     xMean = mean.vec,
                     Xs = muX,
                     fixedX = mu.matrix,
                     randX = random.effectsX,
                     clustRandEffX = clustX,
                     Sigma = Sigma,
                     d = d))
    }else{
        return( final.results )
    }
    
    
    
    
}



