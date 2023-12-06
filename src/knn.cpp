
//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Rnanoflann.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>
using namespace arma;
using namespace Rcpp;

using Rnanoflann::KDTreeArmadilloAdaptor;
using Rnanoflann::KDTreeArmadilloAdaptor2;
using Rnanoflann::KDTreeArmadilloAdaptor3;
using Rnanoflann::KDTreeArmadilloAdaptor4;

template <class T>
void nn_helper(T &mat_index, SearchParameters &searchParams, arma::mat &points, arma::uword k,
               const std::string &search, const double radius, const bool parallel,
               const unsigned int cores, arma::umat &indices, arma::mat &distances)
{
    if (search == "standard")
    {
        if (parallel)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(cores)
#endif
            for (uword i = 0; i < points.n_cols; ++i)
            {
                mat_index.index_->knnSearch(points.colptr(i), k, indices.colptr(i), distances.colptr(i));
            }
        }
        else
        {
            for (uword i = 0; i < points.n_cols; ++i)
            {
                mat_index.index_->knnSearch(points.colptr(i), k, indices.colptr(i), distances.colptr(i));
            }
        }
    }
    else if (search == "radius")
    {
        if (parallel)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(cores)
#endif
            for (uword i = 0; i < points.n_cols; ++i)
            {
                std::vector<ResultItem<uword, double>> radius_search_results; // Store the results
                radius_search_results.reserve(k);
                mat_index.index_->radiusSearch(points.memptr(), radius, radius_search_results);

                uword *ind = indices.colptr(i);
                double *dist = distances.colptr(i);
                for (uword j = 0; j < radius_search_results.size(); ++j)
                {
                    ind[j] = radius_search_results[j].first;
                    dist[j] = radius_search_results[j].second;
                }
            }
        }
        else
        {
            for (uword i = 0; i < points.n_cols; ++i)
            {
                std::vector<ResultItem<uword, double>> radius_search_results; // Store the results
                radius_search_results.reserve(k);
                mat_index.index_->radiusSearch(points.memptr(), radius, radius_search_results);

                uword *ind = indices.colptr(i);
                double *dist = distances.colptr(i);
                for (uword j = 0; j < radius_search_results.size(); ++j)
                {
                    ind[j] = radius_search_results[j].first;
                    dist[j] = radius_search_results[j].second;
                }
            }
        }
    }
}

//[[Rcpp::export]]
List nn(arma::mat data, arma::mat points, arma::uword k, const std::string method = "euclidean",
        const std::string search = "standard", const double eps = 0.0, const bool square = false,
        const bool sorted = false, const double radius = 0.0, const unsigned int leafs = 10, const double p = 0.0,
        const bool parallel = false, const unsigned int cores = 0)
{
    umat indices(k, points.n_cols);
    mat distances(k, points.n_cols);
    SearchParameters searchParams(eps, sorted);

    if (method == "euclidean")
    {
        if (square)
        {
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, Rnanoflann::euclidean, true>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }
        else
        {
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, Rnanoflann::euclidean, false>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }
    }
    else if (method == "hellinger")
    {
        if (square)
        {
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, Rnanoflann::hellinger, true>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }
        else
        {
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, Rnanoflann::hellinger, false>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }
    }
    else if (method == "manhattan")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::manhattan>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "canberra")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::canberra>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "kullback_leibler")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::kullback_leibler>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "jensen_shannon")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor4<mat, Rnanoflann::jensen_shannon>;
        my_kd_tree_t mat_index(data.n_rows, data, points, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "itakura_saito")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::itakura_saito>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "bhattacharyya")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::bhattacharyya>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "jeffries_matusita")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::jeffries_matusita>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "minimum")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::minimum>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "maximum")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::maximum>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "total_variation")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::total_variation>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "sorensen")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::sorensen>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "cosine")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::cosine>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "gower")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::gower>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "minkowski")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor3<mat, Rnanoflann::minkowski>;
        my_kd_tree_t mat_index(data.n_rows, data, p, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "soergel")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::soergel>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "kulczynski")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::kulczynski>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "wave_hedges")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::wave_hedges>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "motyka")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::motyka>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else if (method == "harmonic_mean")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, Rnanoflann::harmonic_mean>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    else
    {
        stop("Unsupported Method: %s", method);
    }

    return List::create(_["indices"] = indices + 1, _["distances"] = distances);
}