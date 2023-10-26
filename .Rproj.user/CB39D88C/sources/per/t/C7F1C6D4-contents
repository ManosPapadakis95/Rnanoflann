\name{nn}
\alias{nn}
\title{k-nearest neighbours search}
\description{
Uses a kd-tree to find the k nearest neighbours for each point in a given dataset.
}
\usage{
nn(data, points = data, k = nrow(data), method = "euclidean", search = "standard", 
eps = 0.0, square = FALSE, sorted = FALSE, radius = 0.0, trans = TRUE, leafs = 10, 
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
The number of neares neighbours to search for.
}
\item{method}{
The type of distance. Currently two distances are supported, the "euclidean" and "manhattan".
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
\item{leafs}{
Number of divided points. Default is 10.

\itemize{
    \item Large values mean that the tree will be built faster (since the tree will be smaller), but each query will be slower (since the linear search in the leaf is to be done over more points).
    \item Small values will build the tree much slower (there will be many tree nodes), but queries will be faster... up to some point, since the "tree-part" of the search (logarithmic complexity) still has a significant cost.
}
}
\item{parallel}{
Should the computations take place in parallel? The default value is FALSE.
}
\item{cores}{
Number of threads for parallel version. The default is 0 which means all the available threads.
}
}

\value{
    A list with 2 fields.
	\itemize{
		\item \bold{inds}: A matrix with the indices of each nearest neighbour for each of the rows of the matrix "points".
		\item \bold{dists}: A matrix with the distances between each nearest neighbour and each of the rows of the matrix "points".
	}
}

\examples{
x <- as.matrix(iris[1:140, 1:4])
xnew <- as.matrix(iris[141:150, 1:4])
nn(data = x, points = xnew, k = 10)
}


