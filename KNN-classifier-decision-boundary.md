KNN classifier decision boundary
================
2DoLLars

K-nearest neighbor classifier has very simple classification rule based
on voting. Unlike parametric classification model such as LDA(Linear
discriminant analysis), there is no parametric assumption on KNN
classifier. Here, we give some examples on KNN classifier and plot its
decision boundary. Function `knn.index` is to find the index of
K-nearest neighbor of given point.

``` r
knn.index = function(x_data, x_predict, K)
{
   # if (!is.matrix(x_predict))
   # {
   #    num_of_predict = 1
   # }else{
   #    num_of_predict = nrow(x_predict)
   # }
   
   index_result = rep(NA, K)
   
   # for (i in 1 : num_of_predict)
   # {
      distance = sqrt(colSums((t(x_data) - x_predict)^2))
      distance_neighbors = distance[distance %in% sort(distance)[1 : K]]
      
      if ( length(distance_neighbors) > K )
      {
         neighbors_max = max(distance_neighbors)
         overlap_idx = sample(which(distance == neighbors_max), 1)
      }
      index = distance %in% distance_neighbors
      index_num = which(index)
      
      if ( length(distance_neighbors) > K ) index_num = index_num[ !index_num == overlap_idx ]
      index_result = index_num
   # }
   
   return(index_result)
}
```

## Scatter plot

The following plot is the scatter plot of iris data using `Sepal.Length`
and `Sepal.Width`.

``` r
data(iris)
x1 = iris$Sepal.Length
x2 = iris$Sepal.Width
x_data = cbind(x1, x2)

K = 5

plot(x1, x2, col = "white",
     main = paste0(K, "-nearst neigbhor decision boundary"),
     xlab = "Sepal.Length",
     ylab = "Sepal.Width")

points(x1[iris$Species == "setosa"], x2[iris$Species == "setosa"],
       col = "red", pch = 2, cex = 1.5, lwd = 2)
points(x1[iris$Species == "versicolor"], x2[iris$Species == "versicolor"],
       col = "blue", pch = 3, cex = 1.5, lwd = 2)
points(x1[iris$Species == "virginica"], x2[iris$Species == "virginica"],
       col = "green", pch = 5, cex = 1.5, lwd = 2)
```

![](/image/knn_scatter_plot.png)<!-- -->

## Decision boundary of KNN classifier

we plot decision boundary of KNN classifier. Since it is non-parametric
method, there needs to form grid to plot decision boundary.

``` r
plot(x1, x2, col = "white",
     main = paste0(K, "-nearst neigbhor decision boundary"),
     xlab = "Sepal.Length",
     ylab = "Sepal.Width")

points(x1[iris$Species == "setosa"], x2[iris$Species == "setosa"],
       col = "red", pch = 2, cex = 1.5, lwd = 2)
points(x1[iris$Species == "versicolor"], x2[iris$Species == "versicolor"],
       col = "blue", pch = 3, cex = 1.5, lwd = 2)
points(x1[iris$Species == "virginica"], x2[iris$Species == "virginica"],
       col = "green", pch = 5, cex = 1.5, lwd = 2)

z1 = seq(min(x1), max(x1), length.out = 50)
z2 = seq(min(x2), max(x2), length.out = 50)

grid = expand.grid(z1, z2)
grid = as.matrix(grid)

class_vec = rep(NA, nrow(grid))

for (i in  1 : nrow(grid))
{
   index = knn.index(x_data, grid[i, ], K)
   class = which.max(table(iris$Species[index]))
   
   class_name = names(class)
   class_vec[i] = class_name
   if( class_name == "setosa" )
   {
      col = "red"
   }else if( class_name == "versicolor") {
      col = "blue"
   }else{
      col = "green"
   }
      
   points(grid[i, ][1], grid[i, ][2], col = adjustcolor(col, 0.2), pch = 20)
}
```

![](/image/knn_decision_boundary.png)<!-- -->
