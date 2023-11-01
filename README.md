# Rnanoflann  ![Rnanoflann](https://raw.githubusercontent.com/jlblancoc/nanoflann/master/doc/logo.png)

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![metacran downloads](https://cranlogs.r-pkg.org/badges/grand-total/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![metacran downloads](https://cranlogs.r-pkg.org/badges/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![CRAN_latest_release_date](https://www.r-pkg.org/badges/last-release/Rnanoflann)](https://cran.r-project.org/package=Rnanoflann) [![R macos](https://github.com/ManosPapadakis95/Rnanoflann/actions/workflows/r.yml/badge.svg)](https://github.com/ManosPapadakis95/Rnanoflann/actions/workflows/r.yml/macos)
https://img.shields.io/github/workflow/status/ManosPapadakis95/Rnanoflann/r.yml/macos



## 1. About
**Rnanoflann** is a wrapper for C++'s library [nanoflan](https://github.com/jlblancoc/nanoflann) which performs nearest neighbors search using kd-trees.

## 2. Usage
You can use the exported `Rnanoflann::nn` function or directly **nanoflan** via **LinkignTo** mechanism.

### 2.1. Rnanoflann
**Rnanoflann** export the function `nn` that performs nearest neighbors search with options:

*  **data** - An `M x d` `matrix` where each of the M rows is a point.
*  **points** - An `N x d` `matrix` that will be queried against data. d, the number of columns, must be the same as data. If missing, defaults to data.
*  **parallel** - uses omp library to perform parallel search for each point. Default is `FALSE`
*  **cores** - the cores that omp will use. Default is zero and it means to automatically compute the numbers of threads.
*  **search** - the supported types are `standard` and `radius`.
*  **eps** - Error bound. Default is `0.0`.
*  **k** - The maximum number of nearest neighbors to compute. The default value is set to the number of rows in data

### 2.2. LinkingTo
Add in Description in LinkingTo section the Rnanoflann and then:

* use **nanoflann** directly. Just `#include "nanoflann.hpp"`. Refer to [nanoflan](https://github.com/jlblancoc/nanoflann) for more details.
* use the `Rnanoflann::nn` via `C++`. Just `#include "Rnanoflann.h"`. The available implemented function are use `Rcpp` and `RcppArmadillo`. For custom matrices you need to implement you own adaptor (see above).
