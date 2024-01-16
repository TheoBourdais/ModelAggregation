add the independant sum kernel

# the new gaussian correlation kernel

$$K_{ij}(x_k,xl)=\exp{-\frac{1}{2}\lVert \frac{x_k}{l_i}-\frac{x_l}{l_j}\rVert^2}$$

seems to be garbage

Fourrier transform based:

$$K_{ij}(x_k,xl)=\sqrt{\frac{2l_il_j}{l_i^2+l_j^2}}\exp{-\frac{1}{l_i^2+l_j^2}\lVert x_k-x_l\rVert^2}$$

# For independant sum

add the zero condition and see impact

# you can create a new coordinate by
- taking the sum
- taking the max? (not sure)

# What I tried
exponential kernel, with:
- independant models
    - Y=X1+X2
    - cov(Y,X1)=Cov(X1,X1), cov(Y,X2)=Cov(X2,X2)
- correlated models Cov(X1,X2)=exp(...)
    - Y=X1+X2
    - cov(Y,X1)=Cov(X1,X1)+Cov(X1,X2), cov(Y,X2)=Cov(X2,X2)+Cov(X2,X1)
None seem to work well