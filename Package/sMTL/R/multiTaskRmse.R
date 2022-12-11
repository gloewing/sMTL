#' multiTaskRmse: RMSE for multi-task problems (averaged across tasks)
#' @param data A matrix
#' @param beta A matrix
#' @export


multiTaskRmse <- function(data, beta){
    K <- length( unique( data$Study ) ) # number of studies
    trainSet <- vector(length = K) # list for indices
    Xindx <- which( ! colnames(data) %in% c("Y", "Study"))
    
    for(j in 1:K){
        indx <- which(data$Study == j) # indices of current studies
        
        # rmse for jth study using jth beta (i.e., jth model)
        trainSet[j] <- sqrt( mean( 
            (  data$Y[indx] - cbind(1, as.matrix(data[indx, Xindx]) ) %*% beta[,j]  )^2 
        ) 
        )
        
    }
    
    
    return( mean(trainSet) )
    
}
