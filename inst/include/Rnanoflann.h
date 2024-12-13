#pragma once

#include "internal/dists.hpp"
#include "internal/KDTreeArmadilloAdaptor.hpp"

List nn(arma::mat data, arma::mat points, arma::uword k, const std::string method = "euclidean",
        const std::string search = "standard", const double eps = 0.0, const bool square = false,
        const bool sorted = false, const double radius = 0.0, const unsigned int leafs = 10, const double p = 0.0,
        const bool parallel = false, const unsigned int cores = 0);