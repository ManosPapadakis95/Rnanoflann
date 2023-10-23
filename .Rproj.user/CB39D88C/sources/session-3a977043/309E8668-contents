
//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "KDTreeArmadilloAdaptor.hpp"
#include <omp.h>
#include <string>
using namespace arma;
using namespace Rcpp;

// [[Rcpp::interfaces(cpp)]]

//[[Rcpp::export]]
List knn(arma::mat data, arma::mat points, arma::uword k,const std::string search = "standard", 
         const double eps = 0.0, const bool sorted = false, const double radius = 0.0, const bool parallel = false)
{
    // Create a MyData object that wraps the Armadillo matrix
    // Create a KD-Tree Adaptor for Armadillo
    using my_kd_tree_t = KDTreeArmadilloAdaptor<mat>;
    my_kd_tree_t mat_index(data.n_rows, data);
    
    umat indices(k,points.n_cols);
    mat distances(k,points.n_cols);
    
    SearchParameters searchParams(eps,sorted);
    
    if(search == "standard"){
        #pragma omp parallel for if(parallel)
        for(uword i=0;i<points.n_cols;++i){
            mat_index.index_->knnSearch(points.colptr(i), k, indices.colptr(i), distances.colptr(i));
        }
    }else if(search == "radius"){
        #pragma omp parallel for if(parallel)
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
    
    return List::create(_["indices"]=indices+1, _["distances"]=distances);
}