\name{nn}
\alias{nn}
\title{Nearest Neighbour Search}
\description{
Uses a kd-tree to find the k nearest neighbours for each point in a given dataset.
}
\usage{
nn(data, points = data, k = nrow(data), search = "standard", eps = 0.0, 
                sorted = FALSE, radius = 0.0, trans = TRUE, parallel = FALSE)
}

\arguments{
\item{data}{A numerical matrix. The k nearest points will be extracted from this matrix.
}
\item{points}{
A numerical matrix. The function will find the nearest points of each row from this matrix. 
}
\item{k}{
The number of neares neibhgours to search for.
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

}
\item{parallel}{
Should the computations take place in parallel? The default value is FALSE.
}
}


\value{
  \code{rcpparma_hello_world()} does not return a value, but displays a
  message to the console.

  \code{rcpparma_outerproduct()} returns a numeric matrix computed as the
  outer (vector) product of \code{x}.

  \code{rcpparma_innerproduct()} returns a double computer as the inner
  (vector) product of \code{x}.

  \code{rcpparma_bothproducts()} returns a list with both the outer and
  inner products.
  
}
\details{
  These are example functions which should be largely
  self-explanatory. Their main benefit is to demonstrate how to write a
  function using the Armadillo C++ classes, and to have to such a
  function accessible from R.
}
\references{
  See the documentation for Armadillo, and RcppArmadillo, for more details.
}

\examples{
  x <- sqrt(1:4)
  rcpparma_innerproduct(x)
  rcpparma_outerproduct(x)
}

\author{Dirk Eddelbuettel}

