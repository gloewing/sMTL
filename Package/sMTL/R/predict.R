#' predict: predict on smtl model object
#'
#' @param model An sMTL model object returned from the smtl() function
#' @param X A matrix of deatures
#' @param stack An optional boolean specifying whether to calculate and apply stacking weights (only for Domain Generalization problems). 
#' @param lambda_1 A optional numeric scalar specifying which lambda_1 to use for prediction. Only needed if the model object is fit on a path (multiple hyperparameterr values)
#' @param lambda_2 A optional numeric scalar specifying which lambda_2 to use for prediction. Only needed if the model object is fit on a path (multiple hyperparameterr values)
#' @param lambda_z A optional numeric scalar specifying which lambda_2 to use for prediction. Only needed if the model object is fit on a path (multiple hyperparameterr values)
#' @return A matrix of task-specific predictions for multi-task/multi-label or for Domain Generalization problems, average and multi-study stacking predictions.
#' @examples
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Installed and Julia Path is Known ######
#' #####################################################################################
#' # fit model
#' mod <- smtl(y = y, 
#'             X = X, 
#'             study = task, 
#'             s = 5, 
#'             commonSupp = FALSE,
#'             lambda_1 = c(0.1, 0.2, 0.3),
#'             lambda_z = c(0.01, 0.05, 0.1))
#' 
#' # make predictions
#' preds <- sMTL::predict(model = mod, 
#'                        X = X, 
#'                        lambda_1 = 0.1, 
#'                        lambda_z = 0.01)

predict = function(model, 
                    X, 
                    lambda_1 = NA, 
                    lambda_2 = NA, 
                    lambda_z = NA,
                    stack = FALSE
    ) {
    
    tuneVector <- c(lambda_1, lambda_2, lambda_z)
    
    # if all tuning values NA and only one model was fit
    if( all( is.na( tuneVector ) ) & nrow(model$grid) == 1){
        tuneVector <- model$grid
    }else if( all( is.na( tuneVector ) ) & nrow(model$grid) > 1){
        # if all tuning values NA and multiple models were fit
        message("Multiple models were fit at different tuning values. \n You must specify the tuning values you want predictions for")
        return(NA)
    }else if( !all( is.na( tuneVector ) ) & sum(is.na(tuneVector)) > 0 & nrow(model$grid) > 1 ){
        # if some but not all tuning values NA and multiple models were fit then
        # set missing tuning values to 0
        
        zero_indx <- which(is.na(tuneVector)) # which tuning values not specified
        tuneVector[zero_indx] <- 0 # set them to 0
    }
    
    
    if(model$reg_type == "L0"){
        #########################
        # individual regression
        #########################
        
        if(nrow(model$grid) > 1){
            # were multiple tuning values used
            
            # which matches tuning values (just the first one)
            indx <- which( apply(model$grid,1, function(x) all(x == tuneVector ) ) )[1]
            b <- model$beta[,indx]
            
            if(is.na(indx))   message("No models fit on tuning values specified")
            
        }else{
            # just a vector
            b <- as.numeric(model$beta)
        }
        
        
        preds <- cbind(1, as.matrix(X) ) %*% b
        
        return(preds)
        
        
    }else if(model$reg_type == "multiLabel" | model$reg_type == "multiStudy" ){
        ##########################
        # multi-task / multi-study
        ##########################
        
        if(nrow(model$grid) > 1){
            # were multiple tuning values used
            
            # which matches tuning values (just the first one)
            indx <- which( apply(model$grid,1, function(x) all(x == tuneVector ) ) )[1]
            b <- model$beta[, ,indx]
            
            if(is.na(indx))   message("No models fit on tuning values specified")
        }else{
            # just a matrix
            b <- model$beta
        }
        
        
        preds <- cbind(1, as.matrix(X) ) %*% b
        
        colnames(preds) <- paste0("task_", 1:model$K)

            if(stack){
                
                message("Fitting Stacking Model")
                
                # nnls for stacking
                fitW <- glmnet::glmnet(y = as.vector(model$y_train),
                                       x = as.matrix( cbind(1, as.matrix(model$X_train) ) %*% b   ),
                                       alpha = 0,
                                       lambda = 0,
                                       standardize = TRUE,
                                       intercept = TRUE,
                                       thresh = 1e-10,
                                       lower.limits = 0)
                
                w <- glmnet::coef.glmnet(fitW)
                rm(fitW)
                
                avg_preds <- rowMeans(preds)
                stack_preds <- cbind(1, preds) %*% w
                
                preds <- cbind(avg_preds, stack_preds  )
                colnames(preds) <- c("avg_predictions", "stack_predictions")
            }

        
        return(preds)
    }
    

}
    