
nn <- function(data, points = data, k = nrow(data), search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, leafs = 10, parallel = FALSE, cores = 0){
    res <- .Call(`_Rnanoflann_nn`, t(data), t(points), k, search, eps, sorted, radius, leafs, parallel, cores)
    if(trans){
        res$indices <- t(res$indices)
        res$distances <- t(res$distances)
    }
    res
}
