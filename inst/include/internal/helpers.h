#pragma once

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

template <class RET, class... Args>
using Mfunction = RET (*)(Args...);

using Binary_Function = Mfunction<const double &, const double &, const double &>;
template <class T>
using ConditionFunction = bool (*)(T);

template <Binary_Function F, class T>
double sum_with(T x, T y)
{
	double a = 0;
	typename T::iterator startx = x.begin();
	typename T::iterator starty = y.begin();
	for (; startx != x.end(); ++startx, ++starty)
	{
		a += F(*startx, *starty);
	}
	return a;
}

template <class T, ConditionFunction<T> COND, class F>
T sum_with_condition(F x)
{
	T a = 0;
	for (typename F::iterator start = x.begin(); start != x.end(); ++start)
	{
		if (COND(*start))
		{
			a += *start;
		}
	}
	return a;
}

template <Binary_Function F>
colvec elems(colvec x, colvec y)
{
	colvec maxs(x.n_elem, fill::none);
	for (unsigned int i = 0; i < x.n_elem; ++i)
	{
		maxs[i] = F(x[i], y[i]);
	}
	return maxs;
}

inline bool check_if_is_finite(double x)
{
	return x > 0 and !R_IsNA(x);
}