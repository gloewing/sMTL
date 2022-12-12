#' reName_cv: rename output from CV. For internal package use only.
#'
#' @param x A list
#' @examples
#' 
#' reName_cv(list(best = testLambda, best.1se = se1, rmse = rmseMat, avg = avg))
#'  @return A list with elements renamed
#' @export

reName_cv <- function(x){
    
    lambda1 <- lambda2 <- rho <- NULL # global variable declaration for CRAN checks
    
    # rename parameters for tuning function
    if(is.object(x$best)){
        x$best <- dplyr::rename(x$best, s = rho)
        if( "lambda1" %in% names(x$best) ){
            x$best <- dplyr::rename(x$best, lambda_1 = lambda1)
        }     
        if( "lambda2" %in% names(x$best) ){
            x$best <- dplyr::rename(x$best, lambda_2 = lambda2) 
        }  
    }
    
    # rename parameters for 1se function
    if(is.object(x$best.1se)){
        x$best.1se <- dplyr::rename(x$best.1se, s = rho)
        if( "lambda1" %in% names(x$best.1se) ){
            x$best.1se <- dplyr::rename(x$best.1se, lambda_1 = lambda1)
        }     
        if( "lambda2" %in% names(x$best.1se) ){
            x$best.1se <- dplyr::rename(x$best.1se, lambda_2 = lambda2) 
        }  
    }
    
    
    # rmse 
    if(is.object(x$rmse)){
        if( "lambda1" %in% rownames(x$rmse) )    rownames(x$rmse)[rownames(x$rmse) == "lambda1"] = "lambda_1"
        if( "lambda2" %in% rownames(x$rmse) )    rownames(x$rmse)[rownames(x$rmse) == "lambda2"] = "lambda_2"
        if( "rho" %in% rownames(x$rmse) )    rownames(x$rmse)[rownames(x$rmse) == "rho"] = "s"
    }
    
    
    # avg 
    if(is.object(x$avg)){
        if( "lambda1" %in% rownames(x$avg) )    rownames(x$avg)[rownames(x$avg) == "lambda1"] = "lambda_1"
        if( "lambda2" %in% rownames(x$avg) )    rownames(x$avg)[rownames(x$avg) == "lambda2"] = "lambda_2"
        if( "rho" %in% rownames(x$avg) )    rownames(x$avg)[rownames(x$avg) == "rho"] = "s"
    }
    
    
    return(x)
}

