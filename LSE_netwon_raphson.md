LSE by Netwon-Raphson method
================
2DoLLars

In numerical analysis, Newton’s method, also known as the Newton–Raphson
method, named after Isaac Newton and Joseph Raphson, is a root-finding
algorithm which produces successively better approximations to the roots
(or zeroes) of a real-valued function. In this post, we give an example
by getting least squares estimate via Newton-Raphson method.

## Gradient vector and hessian

We need the gradient vector and hessian to implement Newton-Raphson
method.

``` r
H = function(n,x)
{
  2 * matrix(c(n, sum(x),
               sum(x), sum(x^2)), nrow = 2, ncol = 2)
}
  
grad = function(b,x,y)
{
   c(-2 * sum(y - b[1] - b[2] * x),
     -2 * sum(x * (y - b[1] - b[2] * x)))
}
```

## Main function

The following function is for implementing Newton-Raphson method for
getting LSE.

``` r
LSE = function(b_old, data, y_idx, x_idx, iter.max = 50)
{
  iter = 0
  n = nrow(data)
  y = data[,y_idx]
  x = data[,x_idx]
  error = 10
  while( (iter <= iter.max) & (sum(abs(error)) > 1e-10) ){
    iter = iter + 1
    b_new = b_old - solve(H(n,x)) %*% grad(b_old,x,y)
    error = b_old - b_new
    b_old = b_new
  }
  return(c(beta = b_old, iter = iter))
}
```

## LSE for iris data

We get two LSEs. One is by Newton-Raphson method and the other is by
lm-fuction in R. It is found that the value of both are the same.

``` r
b_old = c(-10, -7)
LSE_fit = LSE(b_old, iris, 1, 2, iter = 100)
lm = lm(Sepal.Length ~ Sepal.Width, data = iris[,-5])
LSE_coef = LSE_fit[1 : 2]
LSE_coef = round(LSE_coef, digits = 4)
lm_coef = lm$coefficients
lm_coef = round(lm_coef, digits = 4)

print(paste0("LSE by Newton-Raphson method : ", LSE_coef[1], ", ", LSE_coef[2]))
```

    ## [1] "LSE by Newton-Raphson method : 6.5262, -0.2234"

``` r
print(paste0("LSE by lm-function : ", lm_coef[1], ", ", lm_coef[2]))
```

    ## [1] "LSE by lm-function : 6.5262, -0.2234"
