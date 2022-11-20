#' seReturn: find smallest rho within 1 se of smallest cv error 
#' @param x dataframe
#' @export

# 1se rule for selecting rho

seReturn <- function(x){
    
    # pre-process
    e <- x %>% 
        dplyr::as_tibble() %>% 
        t() 
    
    colnames(e) <- rownames(x) # new names because we transposed
    
    minVec <- e %>% 
        dplyr::as_tibble() %>% 
        dplyr::group_by(rho) %>% 
        dplyr::summarise(m = min(avg)) # get the minimum for each rho
    
    sdVal <- sd(minVec$m) # get standard deviation across minima
    
    mindx <- which(minVec$m <= min(minVec$m) + sdVal )[1] # get smallest rho that is within rrange
    rhoStar <- minVec$rho[mindx] # index of  best tuning values for this rho
    
    bestVal <- e %>% 
        dplyr::as_tibble() %>% 
        dplyr::filter(rho == rhoStar) %>% 
        dplyr::slice_min(avg) %>%
        dplyr::select( -avg ) %>%
        as.data.frame()
    
    return(bestVal)
    
}
