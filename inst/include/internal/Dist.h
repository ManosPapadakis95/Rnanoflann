#pragma once
//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

namespace Dist
{
    inline double manhattan(colvec x, colvec y)
    {
        return sum(abs(x - y));
    }

    inline double euclidean(colvec x, colvec y)
    {
        return sum(square(x - y));
    }
}