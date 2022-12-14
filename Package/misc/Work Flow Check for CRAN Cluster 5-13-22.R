library(devtools)
path <- "/n/home12/gloewinger/sMTL"
# for julia call if it doesnt find path
#### find julia path on Julia with command   > println(Sys.BINDIR)

#julia_setup(installJulia = TRUE)
# which will invoke install_julia automatically if Julia is not found and also do initialization of JuliaCall.

# if it doesnt work (cant find julia then specify path and set up)
#julia_setup(install = FALSE, JULIA_HOME = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin") # for juliaCall
# https://cran.r-project.org/web/packages/JuliaCall/readme/README.html

# install packages
#julia_install_package_if_needed("TSVD")
#julia_install_package_if_needed("Statistics")

# a = julia_command("a = Sys.BINDIR;")
# juliaPath = julia_eval("a")
# 
# library(JuliaConnectoR)
# Sys.setenv(JULIA_BINDIR = juliaPath)

#fileConn<-file("/Library/Frameworks/R.framework/Versions/4.0/Resources/library/sMTL/julia_path/julia_path.txt")
fileConn <- file("~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package/sMTL/inst/julia_path/julia.path.txt")
#writeLines("/Applications/Julia-1.5.app/Contents/Resources/julia/bin", fileConn)
writeLines("NA", fileConn) # initially is just NA to signal not changed
close(fileConn)

## used this to create MIT license
# path <- "~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package/sMTL"
# setwd(path)
# use_mit_license("Gabriel Conan Loewinger")

# remove old study strap package
detach("package:sMTL", unload=TRUE)
remove.packages("sMTL")

setwd(path)

# document and build package
devtools::document()
devtools::build()

# install new one
install.packages("/n/home12/gloewinger/sMTL/sMTL_0.1.0.tar.gz", 
                 repos = NULL, type = "source", dependencies = TRUE)

install.packages("sMTL_0.1.0.tar.gz", repos = NULL, type = "source")

library(sMTL)
smtl_setup(path = "/Applications/Julia-1.5.app/Contents/Resources/julia/bin", installJulia = FALSE, installPackages = FALSE)

# document and build package -- Need to repeat this again because need smtl_setup run before vignette can be 
# properly built
devtools::document()
devtools::build()

# check for CRAN
path <- "~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package"
setwd(path)
devtools::check("sMTL")


# package website
library(pkgdown)
usethis::use_pkgdown()
# ✓ Adding '^_pkgdown\\.yml$', '^docs$' to '.Rbuildignore'
# ✓ Adding '^pkgdown$' to '.Rbuildignore'
# ✓ Adding 'docs' to '.gitignore'
# • Record your site's url in the pkgdown config file (optional, but recommended)
# • Modify '_pkgdown.yml'


### Final Check on CRAN servers
setwd("~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package")
devtools::check_win_release("sMTL")
devtools::check_win_devel("sMTL")

## check before submission on your computer
setwd("~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package")
spell_check("~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package/sMTL")
check_rhub("~/Desktop/Research Final/Sparse Multi-Study/CRAN Package/Original Code used to Build Package/sMTL")

### Final Submission to CRAN
setwd("~/Desktop/Research Final/Methods Paper/studyStrap Package/Original Code used to Build Package/studyStrap")
devtools::release()






#### Did on Cluster: # revdepcheck
########## In R on Cluster: only necessary for updates to packages
library(devtools)
library(revdepcheck)
setwd("/n/home12/gloewinger/Original Code used to Build Package")
revdep_check("studyStrap")







# library(crancache)
# revdep_check <- function(pkg = ".",
#                          dependencies = c("Depends", "Imports",
#                                           "Suggests", "LinkingTo"),
#                          quiet = TRUE,
#                          timeout = as.difftime(10, units = "mins"),
#                          num_workers = 1,
#                          bioc = TRUE,
#                          env = revdep_env_vars()) {
#     
#     pkg <- pkg_check(pkg)
#     dir_setup(pkg)
#     if (!db_exists(pkg)) {
#         db_setup(pkg)
#     }
#     
#     did_something <- FALSE
#     repeat {
#         stage <- db_metadata_get(pkg, "todo") %|0|% "init"
#         switch(stage,
#                init =    revdep_init(pkg, dependencies = dependencies, bioc = bioc),
#                install = revdep_install(pkg, quiet = quiet, env = env),
#                run =     revdep_run(pkg, quiet = quiet, timeout = timeout,
#                                     num_workers = num_workers, env = env),
#                report =  revdep_final_report(pkg),
#                done =    break
#         )
#         did_something <- TRUE
#     }
#     
#     if (!did_something) {
#         message(
#             "* See results of previous run in 'revdep/README.md'\n",
#             "* Reset for another run with `revdepcheck::revdep_reset()`"
#         )
#     }
#     
#     invisible()
# }
# 
# revdep_setup <- function(pkg = ".") {
#     pkg <- pkg_check(pkg)
#     status("SETUP")
#     
#     message("Creating directories and database")
#     
#     invisible()
# }
# 
# 
# revdep_init <- function(pkg = ".",
#                         dependencies = c("Depends", "Imports",
#                                          "Suggests", "LinkingTo"),
#                         bioc = TRUE) {
#     
#     pkg <- pkg_check(pkg)
#     pkgname <- pkg_name(pkg)
#     db_clean(pkg)              # Delete all records
#     
#     "!DEBUG getting reverse dependencies for `basename(pkg)`"
#     status("INIT", "Computing revdeps")
#     revdeps <- cran_revdeps(pkgname, dependencies, bioc = bioc)
#     db_todo_add(pkg, revdeps)
#     
#     db_metadata_set(pkg, "todo", "install")
#     db_metadata_set(pkg, "bioc", as.character(bioc))
#     db_metadata_set(pkg, "dependencies", paste(dependencies, collapse = ";"))
#     
#     invisible()
# }
# 
# revdep_install <- function(pkg = ".", quiet = FALSE, env = character()) {
#     pkg <- pkg_check(pkg)
#     pkgname <- pkg_name(pkg)
#     
#     status("INSTALL", "2 versions")
#     
#     dir_create(dir_find(pkg, "old"))
#     dir_create(dir_find(pkg, "new"))
#     
#     ## Install the package itself, both versions, first the CRAN version
#     ## We instruct crancache to only use the cache of CRAN packages
#     ## (to avoid installing locally installed newer versions.
#     "!DEBUG Installing CRAN (old) version"
#     message("Installing CRAN version of ", pkgname)
#     package_name <- pkg_name(pkg)[[1]]
#     
#     with_envvar(
#         c(CRANCACHE_REPOS = "cran,bioc", CRANCACHE_QUIET = "yes", env),
#         with_libpaths(
#             dir_find(pkg, "old"),
#             rlang::with_options(
#                 warn = 2,
#                 install_packages(pkgname, quiet = quiet, repos = get_repos(bioc = TRUE), upgrade = "always")
#             )
#         )
#     )
#     
#     ## Now the new version
#     "!DEBUG Installing new version from `pkg`"
#     message("Installing DEV version of ", pkgname)
#     with_envvar(
#         c(CRANCACHE_REPOS = "cran,bioc", CRANCACHE_QUIET = "yes", env),
#         with_libpaths(
#             dir_find(pkg, "new"),
#             rlang::with_options(
#                 warn = 2,
#                 install_local(pkg, quiet = quiet, repos = get_repos(bioc = TRUE), force = TRUE, upgrade = "always")
#             )
#         )
#     )
#     
#     # Record libraries
#     lib <- library_compare(pkg)
#     utils::write.csv(lib, file.path(dir_find(pkg, "checks"), "libraries.csv"),
#                      row.names = FALSE, quote = FALSE)
#     
#     db_metadata_set(pkg, "todo", "run")
#     invisible()
# }
# 
# 
# revdep_run <- function(pkg = ".", quiet = TRUE,
#                        timeout = as.difftime(10, units = "mins"),
#                        num_workers = 1, bioc = TRUE, env = character()) {
#     
#     pkg <- pkg_check(pkg)
#     pkgname <- pkg_name(pkg)
#     
#     if (!inherits(timeout, "difftime")) {
#         timeout <- as.difftime(timeout, units = "secs")
#     }
#     
#     todo <- db_todo(pkg)
#     status("CHECK", paste0(length(todo), " packages"))
#     start <- Sys.time()
#     
#     state <- list(
#         options = list(
#             pkgdir = pkg,
#             pkgname = pkgname,
#             quiet = quiet,
#             timeout = timeout,
#             num_workers = num_workers,
#             env = env),
#         packages = data.frame(
#             package = todo,
#             state = if (length(todo)) "todo" else character(),
#             stringsAsFactors = FALSE)
#     )
#     
#     run_event_loop(state)
#     end <- Sys.time()
#     
#     status <- report_status(pkg)
#     cat_line(green("OK: "), status$ok)
#     cat_line(red("BROKEN: "), status$broken)
#     cat_line("Total time: ", vague_dt(end - start, format = "short"))
#     
#     db_metadata_set(pkg, "todo", "report")
#     invisible()
# }
# 
# revdep_final_report <- function(pkg = ".") {
#     db_metadata_set(pkg, "todo", "done")
#     status("REPORT")
#     revdep_report(pkg)
# }
# 
# report_exists <- function(pkg) {
#     root <- dir_find(pkg, "root")
#     file.exists(file.path(root, "README.md")) && file.exists(file.path(root, "problems.md"))
# }
# 
# 
# revdep_reset <- function(pkg = ".") {
#     pkg <- pkg_check(pkg)
#     
#     if (exists(pkg, envir = dbenv))
#         rm(list = pkg, envir = dbenv)
#     
#     unlink(dir_find(pkg, "lib"), recursive = TRUE)
#     unlink(dir_find(pkg, "checks"), recursive = TRUE)
#     unlink(dir_find(pkg, "db"), recursive = TRUE)
#     
#     invisible()
# }
# 
# status <- function(title, info = "") {
#     cat_line(rule(left = bold(title), right = info))
# }
# 
