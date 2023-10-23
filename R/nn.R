<<<<<<< HEAD:R/nn.R
nn <- function(data, points = data, k = nrow(data), search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, parallel = FALSE) {
    res <- .Call(`_Rnanoflann_knn`, t(data), t(points), k, search, eps, sorted, radius, parallel)
=======

nn <- function(data, points = data, k = nrow(data), search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, leafs = 10, parallel = FALSE, cores = 0){
    res <- .Call(`_Rnanoflann_nn`, t(data), t(points), k, search, eps, sorted, radius, leafs, parallel, cores)
>>>>>>> 4e5eaced41b68c8e38f0053f614b471ebd09aded:R/knn.R
    if(trans){
        res$indices <- t(res$indices)
        res$distances <- t(res$distances)
    }
    res
}