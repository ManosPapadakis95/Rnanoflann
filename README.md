# Rnanoflann  ![Rnanoflann](https://raw.githubusercontent.com/jlblancoc/nanoflann/master/doc/logo.png)

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![metacran downloads](https://cranlogs.r-pkg.org/badges/grand-total/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![metacran downloads](https://cranlogs.r-pkg.org/badges/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![CRAN_latest_release_date](https://www.r-pkg.org/badges/last-release/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann)


## 1. About
*Rnanoflann* is a wrapper for C++'s library [nanoflan](https://github.com/jlblancoc/nanoflann) which performs nearest neighbors search using kd-trees.

## 2. Usage
You can use the exported `Rnanoflann::nn` function or directly *nanoflan* via *LinkignTo* mechanism.

## 3. Rnanoflann
*Rnanoflann* export the function `nn` that performs nearest neighbors search with options:
 > a. parallel - uses omp library to perform parallel search for each point.
 > b. cores - the cores that omp will use.
 > c. search - the supported types are `standard` and `radius`.
 > d. eps - Error bound
 > e. k - The maximum number of nearest neighbours to compute. The default value is set to the number of rows in data
