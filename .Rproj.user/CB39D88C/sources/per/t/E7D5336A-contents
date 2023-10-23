

knn <- function(data, points, k = nrow(data),search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, parallel = FALSE){
    res <- .Call(`_Rnanoflann_knn`, t(data), t(points), k, search, eps, sorted, radius, parallel)
    if(trans){
        res$indices <- t(res$indices)
        res$distances <- t(res$distances)
    }
    res
}
