K-means
================
2DoLLars

K-means 는 군집분석(clustering analysis)을 수행하기 위한 알고리즘 중에서
가장 흔히 사용되는 알고리즘이다. 이 문서는 K-means 알고리즘의 구현과
실제 데이터에 적용하는 방법에 대해 논의한다.

## K-means 알고리즘 구현

``` r
my_kmeans = function(data, K, iter.max = 50)
{

  mykmeans = function(data, K, iter.max = 50)
  {
    euclid = function(x,y)
    {
      out = sqrt( (sum((x-y)^2)) )
      return(out)
    }
  
  
    data = data.frame(data)
    col_class = sapply(data, class)
    col_class_idx = col_class != "numeric"
    
    if( sum(col_class_idx) != 0 )
    {
      stop("K-means method does not allow non-numerical variables")
    }
    
    data = as.matrix(data)
    n = nrow(data)
    p = ncol(data)
    
    cluster_old = sample(1:K, n, replace = TRUE)
    
    center_old = matrix(nrow = K, ncol = p)
    for(i in 1:K)
    {
      center_old[i,] = colMeans(data[cluster_old == i, , drop = FALSE])
    }
    
    dist = matrix(nrow = n, ncol = K)
    error = 10
    iter = 1
  
    while( (iter <= iter.max) & (error >= 10e-15) )
    {
      iter = iter + 1
      
      for(i in 1:K)
      {
        dist[,i] = apply(data, 1, function(x) euclid(x, center_old[i,]))
      }
      
      cluster_new = apply(dist, 1, which.min)
      
      center_new = matrix(nrow = K, ncol = p)
      for(i in 1:K)
      {
        center_new[i,] = colMeans(data[cluster_new == i, , drop = FALSE])
      }
      
      error = sqrt( sum((center_old - center_new)^2))
      
      center_old = center_new
    }
  
    return(list(cluster_new, iter))
  }

  result = try(mykmeans(data,   K, iter.max))
  while(class(result) != 'list')
  {
    result = try(mykmeans(data, K, iter.max))
  }
  
  return(result)
}
```

## faithful 데이터 분석

`faithful` 데이터는 미국의 옐로스톤(yellow stone) 국립공원에 위치한 Old
faithful 간헐천의 분출에 대해 분출지속시간과 다음 분출까지의 간격을
기록한 것이다. 아래 그림의 각 점은 하나의 분출에 해당한다. 이때, 분출이
크게 (1) 분출지속시간이 길고 다음 분출까지의 간격이 큰 경우와 (2)
분출지속시간이 짧고 다음 분출까지의 간격이 짧은 경우의 두 가지로 나뉨을
짐작할 수 있다.

``` r
data(faithful)
X = faithful

plot(X[,1], X[,2],
     main = "faithful data",
     ylab = "length of Waiting after current eruption",
     xlab = "current eruption's duration")
```

![](K_means_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

이제, K-means 알고리즘을 통해 앞선 짐작이 타당한지 확인해보자. 아래는
K-means 알고리즘을 수행한 결과이다. 데이터가 빨간색(cluster 1)과
파란색(cluster 2)로 잘 나뉘었음을 확인할 수 있다. 즉, 이는 Old faithful
간헐천의 분출이 (1) 과 (2) 의 두 경우로 나뉜다는 주장에 대한 근거가
된다.

``` r
set.seed(2)
faithful_KM = my_kmeans(X, 2)
cluster = faithful_KM[[1]]

plot(X[,1], X[,2],
     col = c("red", "blue")[cluster],
     pch = c(23, 24)[cluster],
     main = "faithful data",
     ylab = "length of Waiting after current eruption",
     xlab = "current eruption's duration")

legend("topleft", c("cluster 1", "cluster 2"),
       pch = c(23,24))
legend("bottomright",
       c("cluster 1", "cluster 2"),
       pch = c("R", "B"),
       col = c("red", "blue"))
```

![](K_means_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
