#' methods names: give name for printing. Internal package use only.
#' @param method A string
#' @param multiLabel A boolean
#' @export

method_nm <- function(method, multiLabel = TRUE){
    
    nm <- ""
    
    if(method == "L0_MS"){
        nm <- "Common Support"
    }else if(method == "L0_MS2"){
        nm <- "Common Support with Beta-Bar Penalty"
    }else if(method == "L0"){
        nm <- "Single L0 Regression"
    }else if(method  == "MS_z3"){
        nm <- "Heterogeneous Support with Separate Active Sets"
    }else if(method  == "MS_z"){
        nm <- "Heterogeneous Support"
    }
    
    if(multiLabel){
        nm <- paste("Multi Label Dataset ||", nm)
    }else{
        nm <- paste("Multi Study Dataset ||", nm)
    }
    
    return(nm)
    
}