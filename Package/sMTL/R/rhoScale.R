#' rhoScale: scale lambda_z depending on magnitude. For internal package use only.
#' @param rhoVec A vector of integers
#' @param itrs An integer
#' @param seed An integer
#' @export


# finds scaling factors for lambda_z based on how big penalty can be
rhoScale <- function(K, p, rhoVec, 
                     itrs = 10000,
                     seed = 1){
    
    # K - number of tasks
    # p - number of covaraites
    # rhoVec is vector of possible rhos
    # is number of random samples
    
    rhoVec <- unique( rhoVec[rhoVec <= p] )# remove any rhos > p
    resMat <- data.frame( matrix(0, nrow = length(rhoVec), ncol = 2) ) # save results
    colnames(resMat) <- c("rho", "scale")
    resMat[,1] <- rhoVec # first column are rhos
    
    for(i in 1:length(rhoVec) ){
        s <- rhoVec[i]
        
        if(s >= p){
            # if s more than number of covariates, make the number big to shrink lambda_z to 0
            
            # exact solution
            resMat[i, 2] <- 1e10
            
        }else if(s * K <= p){
            
            # exact solution
            resMat[i, 2] <- 2 * choose(K, 2) * s / K
            
        }else{
            
            message( paste0("Simulating lambda_z scaling factor for s =", s) )
            # if  s * K > p then simulate

            set.seed(seed)
            resVec <- vector(length = itrs)
            
            vec <- c( rep(0, p - s), rep(1, s) )
            m <- matrix(0, ncol = K, nrow = p)
            
            for(itr in 1:itrs){
                
                # simulate draws
                for(j in 1:K){
                    m[,j] <- sample( vec, replace = FALSE )
                }
                
                # calcualte distance
                resVec[itr] <- sum( dist( t(m) )^2 ) / 4
                
            }
            
            resMat[i, 2] <- max(resVec) # take empirical maximum as approximation to true max
            message("Simulation complete")
        }
        
    }
    
    return(resMat)
}


