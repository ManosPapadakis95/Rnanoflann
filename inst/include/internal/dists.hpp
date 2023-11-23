#pragma once

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "nanoflann.hpp"
#include "Coeff.h"
#include "Dist.h"
#include "helpers.h"
using namespace nanoflann;
using namespace arma;

namespace Rnanoflann
{

    struct euclidean : public Metric
    {
        template <
            class T, class DataSource, bool Square, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct euclidean_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            euclidean_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                DistanceType result = Dist::euclidean(y, x);
                return Square ? result : std::sqrt(result);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, bool Square, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = euclidean_adaptor<T, DataSource, Square, T, IndexType>;
        };
    };

    struct manhattan : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct manhattan_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            manhattan_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                DistanceType result = Dist::manhattan(y, x);
                return result;
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = manhattan_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct hellinger : public Metric
    {
        template <
            class T, class DataSource, bool Square, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct hellinger_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            hellinger_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                DistanceType result = Dist::euclidean(y, x);
                return Square ? result * 0.5 : std::sqrt(result) * (1.0 / std::sqrt(2.0));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, bool Square, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = hellinger_adaptor<T, DataSource, Square, T, IndexType>;
        };
    };

    struct minkowski : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct minkowski_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            minkowski_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                DistanceType result = DistanceType();
                for (size_t i = 0; i < size; ++i)
                {
                    result += pow(abs(a[i] - data_source.kdtree_get_pt(b_idx, i)), data_source.getP());
                }
                return pow(result, data_source.getP_1());
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = minkowski_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct maximum : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct maximum_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            maximum_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                Col<DistanceType> result = abs(x - y);
                return result[result.index_max()];
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = maximum_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct minimum : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct minimum_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            minimum_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                Col<DistanceType> result = abs(x - y);
                return result[result.index_min()];
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = minimum_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct total_variation : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct total_variation_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            total_variation_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return Dist::manhattan(y, x) * 0.5;
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = total_variation_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct cosine : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct cosine_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            cosine_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return dot(y, x) / (sqrt(sum(square(x))) * sqrt(sum(square(y))));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = cosine_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct gower : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct gower_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            gower_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return Dist::manhattan(y, x) * (1.0 / size);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = gower_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct sorensen : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct sorensen_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            sorensen_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return sum(abs(x - y) / (y + x));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = sorensen_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct canberra : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct canberra_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            canberra_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return sum(abs(x - y) / (abs(x) + abs(y)));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = canberra_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct kullback_leibler : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct kullback_leibler_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            kullback_leibler_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                DistanceType res = DistanceType();
                for (size_t i = 0; i < size; ++i)
                {
                    DistanceType y = a[i], x = data_source.kdtree_get_pt(b_idx, i);
                    DistanceType v = (y - x) * (std::log(y) - std::log(x));
                    if (std::isfinite(v))
                    {
                        res += v;
                    }
                }
                return res;
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = kullback_leibler_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct jensen_shannon : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct jensen_shannon_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            jensen_shannon_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                Col<DistanceType> v = x + y;
                return sum_with_condition<double, check_if_is_finite, colvec>(
                    data_source.col_xlogx(b_idx) + data_source.col_ylogy(a) - (arma::log(v) + data_source.log0_5()) % (v));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = jensen_shannon_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct itakura_saito : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct itakura_saito_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            itakura_saito_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                DistanceType res = DistanceType();
                for (size_t i = 0; i < size; ++i)
                {
                    DistanceType y = a[i], x = data_source.kdtree_get_pt(b_idx, i);
                    DistanceType v = (x / y) - (std::log(x) - std::log(y)) - 1.0; // Not symmetric so x/y is not y/x
                    if (std::isfinite(v))
                    {
                        res += v;
                    }
                }

                return res;
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = itakura_saito_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct bhattacharyya : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct bhattacharyya_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            bhattacharyya_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return -log(Coeff::bhattacharyya(y, x));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = bhattacharyya_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct jeffries_matusita : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct jeffries_matusita_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            jeffries_matusita_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return std::sqrt(2.0 - 2.0 * Coeff::bhattacharyya(y, x));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = jeffries_matusita_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct soergel : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct soergel_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            soergel_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return Dist::manhattan(y, x) / sum_with<std::max, Col<DistanceType>>(y, x);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = soergel_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct kulczynski : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct kulczynski_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            kulczynski_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return Dist::manhattan(y, x) / sum_with<std::min, Col<DistanceType>>(y, x);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = kulczynski_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct wave_hedges : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct wave_hedges_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            wave_hedges_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return sum(abs(y - x) / elems<std::max>(y, x));
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = wave_hedges_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct motyka : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct motyka_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            motyka_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return 1.0 - sum_with<std::min, Col<DistanceType>>(y, x) / sum(y + x);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = motyka_adaptor<T, DataSource, T, IndexType>;
        };
    };

    struct harmonic_mean : public Metric
    {
        template <
            class T, class DataSource, typename _DistanceType = T,
            typename IndexType = uint32_t>
        struct harmonic_mean_adaptor
        {
            using ElementType = T;
            using DistanceType = _DistanceType;

            const DataSource &data_source;

            harmonic_mean_adaptor(const DataSource &_data_source)
                : data_source(_data_source)
            {
            }

            DistanceType evalMetric(
                const T *a, const IndexType b_idx, size_t size) const
            {
                Col<DistanceType> y(const_cast<T *>(a), size, false);
                const Col<DistanceType> x = data_source.col(b_idx);
                return 2.0 * dot(y, x) / sum(y + x);
            }

            template <typename U, typename V>
            DistanceType accum_dist(const U a, const V b, const size_t) const
            {
                return 0;
            }
        };

        template <class T, class DataSource, typename IndexType = uint32_t>
        struct traits
        {
            using distance_t = harmonic_mean_adaptor<T, DataSource, T, IndexType>;
        };
    };

};