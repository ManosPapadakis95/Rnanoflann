\name{nn}
\alias{nn}
\title{k-nearest neighbours search}
\description{
Uses a kd-tree to find the k nearest neighbours for each point in a given dataset.
}
\usage{
nn(data, points = data, k = nrow(data), search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, parallel = FALSE)
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
\item{standard}{
The type of search. Apart from the "standard" there are two more options. The "priority" that 
visits cells in increasing order of distance from the query point, and hence, should converge 
more rapidly on the true nearest neighbour, but standard is usually faster for exact searches. 
The "radius" searches only for neighbours within a specified radius of the point. If there 
are no neighbours then the value "indices" will contain 0 and nn.dists will contain 
1.340781e+154 for that point.
}
\item{eps}{
The accuracy of the search. When this is equal to 0, the function will return the exact k 
neighbours. If higher values are supplied, the function will return k approximate neighbours.
}
\item{sorted}{

}
\item{radius}{
The radius of the search, when search="radius".
}
\item{trans}{
Should the return matrices be transposed? The default value is TRUE.
}
\item{parallel}{
Should the computations take place in parallel? The default value is FALSE.
}
}

\value{
\code{indices}{
A matrix with the indices of each nearest neighbour for each of the rows of the matrix "points".
}
\item{distances}{
A matrix with the Euclidean distances between each nearest neighbour and each of the rows of the matrix "points".
}
}

%\details{
%}

%\references{
%See the documentation for Armadillo, and RcppArmadillo, for more details.
%}

\examples{
x <- as.matrix(iris[1:140, 1:4])
xnew <- as.matrix(iris[141:150, 1:4])
nn(data = x, points = xnew, k = 10)
}

\author{Dirk Eddelbuettel}

