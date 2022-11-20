#' tuneZscale: scale lambda_z depending on magnitude. For internal package use only.
#' @param tune.grid A dataframe
#' @param rhoScale A dataframe
#' @export



# takes in tune grid and the output of function rhoScale and re-scales the lambda_z
tuneZscale <- function(tune.grid, rhoScale){
    
    rhoVec <- unique(tune.grid$s) # unique rows
    
    for(rho in rhoVec ){
        
        indx <- which(tune.grid$s == rho) # indices of tune.grid with current rho value
        zScale <- rhoScale$scale[rhoScale$s == rho] # find scaling factor associated with this rho in 
        
        tune.grid$lambda_z[indx] <- tune.grid$lambda_z[indx] / zScale
        
    }
    
    return(tune.grid)
    
}

