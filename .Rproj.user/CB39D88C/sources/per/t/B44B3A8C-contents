
//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Rnanoflann.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>
using namespace arma;
using namespace Rcpp;

using namespace Rnanoflann;

template <class T>
void nn_helper(T& mat_index, SearchParameters &searchParams, arma::mat &points, arma::uword k, 
                const std::string& search, const double radius, const bool parallel, 
                const unsigned int cores, arma::umat &indices, arma::mat &distances){
    if(search == "standard"){
        #ifdef _OPENMP
        #pragma omp parallel for if(parallel) num_threads(cores)
        #endif
        for(uword i=0;i<points.n_cols;++i){
            mat_index.index_->knnSearch(points.colptr(i), k, indices.colptr(i), distances.colptr(i));
        }
    }else if(search == "radius"){
        #ifdef _OPENMP
        #pragma omp parallel for if(parallel) num_threads(cores)
        #endif
        for(uword i=0;i<points.n_cols;++i){
            std::vector<ResultItem<uword, double>> radius_search_results; // Store the results
            radius_search_results.reserve(k);
            mat_index.index_->radiusSearch(points.memptr(), radius, radius_search_results);
            
            uword* ind = indices.colptr(i);
            double* dist = distances.colptr(i);
            for(uword j=0;j<radius_search_results.size();++j){
                ind[j] = radius_search_results[j].first;
                dist[j] = radius_search_results[j].second;
            }
        }
    }
}


//[[Rcpp::export]]
List nn(arma::mat data, arma::mat points, arma::uword k,const std::string method = "euclidean", 
        const std::string search = "standard", const double eps = 0.0, const bool square = false, 
        const bool sorted = false, const double radius = 0.0, const unsigned int leafs = 10, 
        const bool parallel = false, const unsigned int cores = 0)
{
    umat indices(k,points.n_cols);
    mat distances(k,points.n_cols);
    SearchParameters searchParams(eps,sorted);
    
    if (method == "euclidean")
    {
        if(square){
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, euclidean,true>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }else{
            using my_kd_tree_t = KDTreeArmadilloAdaptor2<mat, euclidean,false>;
            my_kd_tree_t mat_index(data.n_rows, data, leafs);
            nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
        }
    }
    else if (method == "manhattan")
    {
        using my_kd_tree_t = KDTreeArmadilloAdaptor<mat, nanoflann::metric_L1>;
        my_kd_tree_t mat_index(data.n_rows, data, leafs);
        nn_helper(mat_index, searchParams, points, k, search, radius, parallel, cores, indices, distances);
    }
    
    return List::create(_["indices"]=indices+1, _["distances"]=distances);
}