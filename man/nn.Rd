\name{nn}
\alias{nn}
\title{k-nearest neighbours search}
\description{
Uses a kd-tree to find the k nearest neighbours for each point in a given dataset.
}
\usage{
nn(data, points = data, k = nrow(data), method = "euclidean", search = "standard", 
eps = 0.0, square = FALSE, sorted = FALSE, radius = 0.0, trans = TRUE, leafs = 10 p = 0.0, 
parallel = FALSE, cores = 0)
}

\arguments{
\item{data}{
A numerical matrix. The k nearest points will be extracted from this matrix.
}
\item{points}{
A numerical matrix. The function will find the nearest neighbours of each row of this matrix. 
}
\item{k}{
The number of nearest neighbours to search for.
}
\item{method}{
The type of distance.See details for the supported metrics.
}
\item{search}{
The type of search. Apart from the "standard" there is the "radius" option. It searches 
only for neighbours within a specified radius of the point. If there are no neighbours 
then the value "indices" will contain 0 and distances will contain 1.340781e+154 for that point.
}
\item{eps}{
The accuracy of the search. When this is equal to 0, the function will return the exact k 
neighbours. If higher values are supplied, the function will return k approximate neighbours.
}
\item{square}{
If you choose "euclidean" as the method, then you can have the option to return the 
squared Euclidean distances by setting this argument to TRUE. Default is FALSE.
}
\item{sorted}{
Should the distances be sorted? This works only when search = "radius".
}
\item{radius}{
The radius of the search, when search = "radius".
}
\item{trans}{
Should the return matrices be transposed? The default value is TRUE.
}
\item{p}{
This is for the the Minkowski, the power of the metric. 
}
\item{leafs}{
Number of divided points. Default is 10.

Large values mean that the tree will be built faster (since the tree will be smaller), 
but each query will be slower (since the linear search in the leaf is to be done over more points).

Small values will build the tree much slower (there will be many tree nodes), but queries 
will be faster... up to some point, since the "tree-part" of the search (logarithmic complexity) 
still has a significant cost.
}
\item{parallel}{
Should the computations take place in parallel? The default value is FALSE.
}
\item{cores}{
Number of threads for parallel version. The default is 0 which means all the available threads.
}
}

\details{
The target of this function is to calculate the distances between xnew and x without having to calculate the whole 
distance matrix of xnew and x. The latter does extra calculations, which can be avoided.

\itemize{
\item euclidean : \eqn{ \sum \sqrt( \sum | P_i - Q_i |^2)}
\item manhattan : \eqn{ \sum \sum | P_i - Q_i |}
\item minimum : \eqn{ \sum \min | P_i - Q_i |}
\item maximum : \eqn{ \sum \max | P_i - Q_i |}
\item minkowski : \eqn{ \sum ( \sum | P_i - Q_i |^p)^\frac{1}{p}}
\item bhattacharyya : \eqn{ \sum - ln \sum \sqrt(P_i * Q_i)}
\item hellinger : \eqn{ \sum 2 * \sqrt( 1 - \sum \sqrt(P_i * Q_i))}
\item kullback_leibler : \eqn{ \sum \sum P_i * log(\frac{P_i}{Q_i})}
\item jensen_shannon : \eqn{ \sum 0.5 * ( \sum P_i * log(2 * \frac{P_i}{Q_i} + Q_i) + \sum Q_i * log(2 * \frac{Q_i}{P_i} + Q_i))}
\item canberra : \eqn{ \sum \sum \frac{| P_i - Q_i |}{P_i + Q_i}}
\item chi_square \eqn{X}^2 : \eqn{ \sum \sum (\frac{(P_i - Q_i )^2}{P_i + Q_i})}
\item soergel : \eqn{ \sum \frac{\sum | P_i - Q_i |}{\sum \max(P_i , Q_i)}}
\item sorensen : \eqn{ \sum \frac{\sum | P_i - Q_i |}{\sum (P_i + Q_i)}}
\item cosine : \eqn{ \sum \frac{\sum (P_i * Q_i)}{\sqrt(\sum P_i^2) * \sqrt(\sum Q_i^2)}}
\item wave_hedges : \eqn{ \sum \frac{\sum | P_i - Q_i |}{\max(P_i , Q_i)}}
\item motyka : \eqn{ \sum \sum \frac{\min(P_i , Q_i)}{(P_i + Q_i)}}
\item harmonic_mean : \eqn{ \sum 2 * \frac{\sum P_i * Q_i}{P_i + Q_i}}
\item jeffries_matusita : \eqn{ \sum \sqrt( 2 - 2 * \sum \sqrt(P_i * Q_i))}
\item gower : \eqn{ \sum \frac{1}{d} * \sum | P_i - Q_i |}
\item kulczynski : \eqn{ \sum \frac{\sum | P_i - Q_i |}{\sum \min(P_i , Q_i)}}
\item itakura_saito : \eqn{ \sum \frac{P_i}{Q_i}  - log(\frac{P_i}{Q_i}) - 1}
}
}

\value{
A list with 2 fields.
\item{indices}{
A matrix with the indices of each nearest neighbour for each of the rows of the matrix "points".
}
\item{distances}{
A matrix with the distances between each nearest neighbour and each of the rows of the matrix "points".
}
}

\examples{
x <- as.matrix(iris[1:140, 1:4])
xnew <- as.matrix(iris[141:150, 1:4])
nn(data = x, points = xnew, k = 10)
}


