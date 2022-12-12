#' smtl_setup: setup Julia path and/or install Julia or Julia packages
#'
#' @param path A string
#' @param installJulia A boolean.
#' @param installPackages A boolean.
#' @return A message
#' @examples
#' 
#' ##################################################################
#' # First Time Loading, Julia is Installed and Julia Path is Known 
#' ##################################################################
#' smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", 
#'            installJulia = FALSE, 
#'            installPackages = FALSE)"
#' 
#' #####################################################################################
#' # If you have run smtl_setup() before, then path specification shouldn't be necessary
#' #####################################################################################
#' smtl_setup(path = NULL, installJulia = FALSE, installPackages = FALSE)"
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Not Installed   ######
#' #####################################################################################
#' smtl_setup(path = NULL, installJulia = TRUE, installPackages = FALSE)"
#' 
#' #####################################################################################
#' ##### First Time Loading, Julia is Installed But Packages NEED INSTALLATION  ######
#' #####################################################################################
#' smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", 
#'            installJulia = TRUE, 
#'            installPackages = TRUE)"
#' @import JuliaCall
#' @export


smtl_setup = function(path = NULL, installJulia = FALSE, installPackages = FALSE) {
    
    # use code from XRJulia to find Julia path
    if(!installJulia & is.null(path)){
        
        # look at path variable in julia_path folder in package for path
        
        # find path julia file
        smtl_path <- paste0( .libPaths("sMTL"), "/sMTL/julia_path/" )
        
        # read julia path
        julia_path <- as.character( utils::read.table(paste0(smtl_path, "julia.path.txt"),  stringsAsFactor = FALSE ) )
        
        if(julia_path == "NA"){
            # if no path given and julia is not to be installed and the path has not been updated then use code from XRJulia to find path
            findJulia <- function(test = FALSE) {
                ## See if a location for the Julia executable has been specified
                ## environment variables JULIA_BIN or JULIA_SRC
                envvar <- Sys.getenv("JULIA_BIN")
                if(!nzchar(envvar)) {
                    src <- Sys.getenv("JULIA_SRC")
                    if(nzchar(src)) {
                        ## try either the standard Julia source, or the directory above bin, etc.
                        trybin <- paste(src, "usr","bin","julia", sep=.Platform$file.sep)
                        if(file.exists(trybin))
                            envvar <- trybin
                        else {
                            trybin <- paste(src, "julia", sep=.Platform$file.sep)
                            if(file.exists(trybin))
                                envvar <- trybin
                        }
                    }
                } # if none of these succeeds, `which julia` used to find an executable version
                if(!nzchar(envvar)) {
                    command <-if (.Platform$OS.type == "windows") "where" else "which"
                    envvar <- tryCatch(system2(command, "julia", stdout = TRUE), warning = function(e) "")
                    if(test)
                        Sys.setenv(JULIA_BIN = envvar) # so next call finds this immediately
                    else if(!nzchar(envvar))
                        stop("No julia executable in search path and JULIA_BIN environment variable not set")
                }
                if(test)
                    nzchar(envvar)
                else
                    envvar
            }
            
            # julia_path <- findJulia() # use this to find path
            # Sys.setenv(JULIA_BINDIR = julia_path )
            
            return( message(paste0("Julia Path Not Specified", #  So We Tried to Find It. Julia Path Found: ", julia_path, 
                                   "\n If incorrect, then do one of the following: ", 
                                   "\n 1) Install Julia by reruning smtl_setup() and set installJulia = TRUE \n 
                                   2) Rerun smtl_setup() and specify path: smtl_setup(installJulia = FALSE, path = <path>) \n
                                   --Run the following in Julia to find the path name: > println(Sys.BINDIR) and specify the path in 2)") ) )
        }else{
            
            # if path works
            Sys.setenv(JULIA_BINDIR = julia_path )
            return( message(paste0("Julia Path Loaded Successfully: No Installation of Julia or Packages Necessary \n
                            Julia Path: ", julia_path) ) )
            
        }
        
    }
  
    # if install=TRUE, use JuliaCall to install
    if(installJulia){
        JuliaCall::julia_setup(installJulia = TRUE, path = path) # which will invoke install_julia automatically if Julia is not found and also do initialization of JuliaCall.
        
        # install packages
        JuliaCall::julia_install_package_if_needed("TSVD")
        JuliaCall::julia_install_package_if_needed("Statistics")
        JuliaCall::julia_install_package_if_needed("LinearAlgebra")
        
        # find the path for the future
        a <- JuliaCall::julia_command("a = Sys.BINDIR;")
        juliaPath <- JuliaCall::julia_eval("a")
        
        # find path of sMTL package
        smtl_path <- paste0( .libPaths("sMTL"), "/sMTL/julia_path/" )
        
        # save Julia path for the future in sMTL package directory for future use
        fileConn <- file( paste0(smtl_path, "julia.path.txt") )
        base::writeLines(juliaPath, fileConn) # save new julia path
        close(fileConn)
        
        # close Julia session
        detach("package:JuliaCall", unload=TRUE) # unload JuliaCall package
        Sys.setenv(JULIA_BINDIR = juliaPath ) # set path for JuliaConnectoR
        
        message("Julia and Julia Packages Installed and Julia Path Found")
    }else{
      
      # if Julia is not to be installed but packages are
      if(installPackages){
        
        message("Installing Julia Packages, Make Sure Path is Correctly Specified")
        
        JuliaCall::julia_setup(installJulia = FALSE, JULIA_HOME = path) # which will invoke install_julia automatically if Julia is not found and also do initialization of JuliaCall.
        
        # install packages
        JuliaCall::julia_install_package_if_needed("TSVD")
        JuliaCall::julia_install_package_if_needed("Statistics")
        
        message("Julia Packages Installed")
        
        detach("package:JuliaCall", unload=TRUE) # unload JuliaCall package
        Sys.setenv(JULIA_BINDIR = path ) # set path for JuliaConnectoR
        
      }
    }
        
    # if path given
    if( !installJulia & !is.null(path) ){
        Sys.setenv(JULIA_BINDIR = path ) # set path for JuliaConnectoR
      
      # find path of sMTL package
      smtl_path <- paste0( .libPaths("sMTL"), "/sMTL/julia_path/" )
      
      # save Julia path for the future in sMTL package directory for future use
      fileConn <- file( paste0(smtl_path, "julia.path.txt") )
      base::writeLines(path, fileConn) # save new julia path
      close(fileConn)
      
      return( message(paste0("Julia Path Loaded Successfully: No Installation of Julia Necessary \n
                            Julia Path: ", path) ) )
    }
    
    
    # if it doesnt work (cant find julia then specify path and set up)
    # julia_setup(JULIA_HOME = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin") # for JuliaCall
    # https://cran.r-project.org/web/packages/JuliaCall/readme/README.html
    
    # install packages
    #julia_install_package_if_needed("TSVD")
    #julia_install_package_if_needed("Statistics")
    
    # a = julia_command("a = Sys.BINDIR;")
    # juliaPath = julia_eval("a")
    # 
    # library(JuliaConnectoR)
    # Sys.setenv(JULIA_BINDIR = juliaPath)
    
    }