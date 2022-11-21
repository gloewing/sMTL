# calculate multi-task rmses with different functions of Y

# multi task RMSE for multi-task (same design matrix)
multiTaskRmse_MT_transform <- function(data, 
                             fn = "raw",
                             Y_orig, # original y vector
                             lambda = NA, # lambda from box cox transform
                             beta,
                             SD = 1){
    
    studyIndx <- which(colnames(full) == "Study")
    XY <- substr(colnames(data), 1, 1) # extract first letter and see which one is Y
    Yindx <- which(XY == "Y") 
    Xindx <- seq(1, ncol(full) )[-c(studyIndx, Yindx)]
    
    K <- length(unique(full$Study))
    
    trainSet <- vector(length = K) # list for indices
    if(fn == "raw"){
        for(j in 1:K){
            
            y_j <- cbind(1, as.matrix( data[, Xindx ] ) ) %*% beta[,j]
            
            outcomeIndx <- Yindx
            
            # rmse for jth study using jth beta (i.e., jth model)
            trainSet[j] <- sqrt( mean( 
                (  data[, outcomeIndx ] - y_j * SD  )^2 
            ) 
            )
            
        }
    }else if(fn == "log"){
        
        for(j in 1:K){
            
            y_j <- cbind(1, as.matrix( data[, Xindx ] ) ) %*% beta[,j]
            
            outcomeIndx <- Yindx
            
            # rmse for jth study using jth beta (i.e., jth model)
            trainSet[j] <- sqrt( mean( 
                (  data[, outcomeIndx ] - (exp(y_j) - 1e-6 ) * SD )^2 
            ) 
            )
            
        }
        
    }else if(fn == "log_quant"){
        
        for(j in 1:K){
            
            ids <- which(full$Study == j)
            constant <- quantile(Y_orig[ids])[4] / quantile(Y_orig[ids])[2]
            
            y_j <- cbind(1, as.matrix( data[, Xindx ] ) ) %*% beta[,j]
            
            outcomeIndx <- Yindx
            
            # rmse for jth study using jth beta (i.e., jth model)
            trainSet[j] <- sqrt( mean( 
                (  data[, outcomeIndx ] - (exp(y_j) - constant ) * SD )^2 
            ) 
            )
            
        }
        
    }else if(fn == "yj"){
        for(j in 1:K){
            
            ids <- which(full$Study == j)

            y_j <- cbind(1, as.matrix( data[, Xindx ] ) ) %*% beta[,j]
            
            outcomeIndx <- Yindx
            
            # rmse for jth study using jth beta (i.e., jth model)
            trainSet[j] <- sqrt( mean( 
                (  data[, outcomeIndx ] - yj_inverse(y = y_j, lambda = lambda[j]) * SD )^2 
            ) 
            )
            
        }
        
        
    }
    
    
    
    return( mean(trainSet) )
    
}

# inverse yeo johnson transform
yj_inverse <- function(y, lambda = 1){
    # assumes y is non-negative here, 
    
    if(lambda == 0){
        y_hat <- (y * lambda + 1)^(1 / lambda) - 1
    }else if(lambda != 0){
        y_hat <- exp(y) - 1
    }
        
        return(y_hat)
    
}
