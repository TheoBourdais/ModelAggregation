We develop a new method that relies on the following observations: If 
$$M(x)=\begin{pmatrix}1 \\ \vdots \\ 1\end{pmatrix} y(x) + \epsilon$$
where
$$\epsilon \sim \mathcal{N}(0,A(x))$$

Then the estimate of $y(x)$ is given by
$$\hat{y}(x) = \frac{\begin{pmatrix}1 \\ \vdots \\ 1\end{pmatrix}^TA^{-1}(x)M(x)}{\begin{pmatrix}1 \\ \vdots \\ 1\end{pmatrix}^TA^{-1}(x)\begin{pmatrix}1 \\ \vdots \\ 1\end{pmatrix}}$$

If one believes that $E[M_i(x)]\neq y(x)$, then one can estimate it by taking $\epsilon\sim\mathcal{N}(\mu(x),A(x))$.

This suggests that we should aggregate by recovering precision matrices! To simplify the recovery, we choose $P$ s.t. $P^TP=I$ and define $A^{-1}(x)=P^T\Lambda(x)P$ where $\Lambda(x)$ is a diagonal matrix. We now that the precision matrix $A^{-1}$ can be recovered by optimizing the following problem:
$$\min_{\Theta}\ (M(x)-y(x))^T\Theta (M(x)-y(x)) + \log|\Theta|$$
Thus, if we fit a Gaussian process for this problem, we get 
$$\min_{\Lambda\in\mathcal{H}} \sum_{i=1}^n (M_i(x)-y(x))^TP\Lambda(x_i)P(M_i(x)-y(x)) + \sum_{k=1}^p\log|\Lambda_k(x_i)|+\lambda \lVert \Lambda \rVert^2_{\mathcal{H}}$$
where $\mathcal{H}$ is a RKHS. 

This presents a problem as the optimization is hard to solve. Moreover, the precision matrix is difficult to approximate. If the error is 0 at a point, then the covariance matrix is singular, making the precision matrix infinite. To tackle these issues, 
- we will use $A^{-1}(x)=P^Te^{\Lambda(x)}P$ where $\Lambda(x)$ is a diagonal matrix and Gaussian process. This allows the precision to evolve rapidly (espicially when the error is low), and also symetrizes the problem between recovering the precision and the covariance matrix. 
- Since precision matrix and covariance matrix is now a symmetric estimation problem, we will instead approximate the covariance matrix 
$$\min_{\Lambda\in\mathcal{H}} \sum_{i=1}^n \lVert P^Te^{-\Lambda(x_i)}P  -(M(x_i)-y(x_i))(M(x_i)-y(x_i))^T\rVert^2_F +\lambda \lVert \Lambda \rVert^2_{\mathcal{H}}$$

This is a regularized version of the constant estimate, which satisfies for $e_i=M(x_i)-y(x_i)$
$$\hat{A}=\min_{A} \sum_{i=1}^n \lVert A  -e_ie_i^T\rVert^2_F$$
where $\hat{A}$ is the covariance matrix.

Choice of $P$:
-----
- If we take $P=I$, then each model is estimated independantly, and we are performing a convex combination
- I suggest to take $P$ as the eigenvectors of the covariance matrix if we want to choose a different $P$. Specifically, this allows to account for the correlation between the models.

Overfitting estimation:
-----
We can model the overfitting by adding a multiplicative term to the errors when the data was used in training. We can define the overfitting coefficient : 
$$C(x)=\begin{pmatrix} e^{c_1} & &0 \\  &\ddots& \\0 & & e^{c_p}   \end{pmatrix}\delta_{training}(x)\text{ where }\delta_{training}(x)= \begin{cases} 1 & \text{if } x\in \text{training set} \\ 0 & \text{otherwise} \end{cases}$$
where the $c_i$ will be learned. Our new loss is then 
$$\min_{\Lambda\in\mathcal{H}, c_i} \sum_{i=1}^n \lVert P^Te^{-\Lambda(x_i)}P  -C(x_i)e_ie_i^TC(x_i)^T\rVert^2_F +\lambda \lVert \Lambda \rVert^2_{\mathcal{H}}$$

>In the case where $P=I$, and we fit each GP independantly, this is like adding a white noise like-kernel to the GP 
>$$\tilde{k}(x,y)=k(x,y)+c\ \delta_{training}(x)\delta(x,y)$$
>where $\delta(x,y)=\begin{cases} 1 & \text{if } x=y \\ 0 & \text{otherwise} \end{cases}$

If the GPs are independant, define $E_i(C)=PC(x_i)e_ie_i^TC(x_i)^TP^T$, we have 

$$\min_{\Lambda\in\mathcal{H}, c_i} \sum_{k=1}^p\left[\sum_{i=1}^n ( e^{-\Lambda_k(x_i)} -[E_i(C)]_{kk})^2_2 +\lambda \lVert \Lambda_k \rVert^2_{\mathcal{H}}\right]$$

One may notice that $(e^{\Lambda_k(x_i)}-Y_i)^2=(\exp(\Lambda_k(x_i)-\log(Y_i))-1)^2Y_i^2\approx (\Lambda_k(x_i)-\log(Y_i))^2Y_i^2+\mathcal{O}((\Lambda_k(x_i)-\log(Y_i))^4)$

So the linearised version of the problem is
$$\min_{\Lambda\in\mathcal{H}, c_i} \sum_{k=1}^p\left[\sum_{i=1}^n ( \Lambda_k(x_i)-\log([E_i(C)]_{kk}))^2[E_i(C)]_{kk}^2 +\lambda \lVert \Lambda_k \rVert^2_{\mathcal{H}}\right]$$

For one independant GP, we define $D=Diag([E_i(C)]_{kk})$, and we have
$$\Lambda_k(x)=k(x,X)D(DKD+\lambda I)^{-1}D\log([E_i(C)]_{kk})$$

>To show this, we use the identity $(P^{-1}+B^TR^{-1}B)^{-1}B^TR^{-1}=PB^T(BPB^T+R)^{-1}$\
>Another version is $V=(K+\lambda/D^2)^{-1}\log([E_i(C)]_{kk})$ but it is highly unstable. 

We can use the linearised version as the final solution or an initialisation for the non-linear problem.

## Dumb version

We can also directly fit the log errors with a GP by minimizing 
$$\min_{\Lambda\in\mathcal{H}} \sum_{i=1}^n \lvert \log([E_i(C)]_{kk})  -\Lambda(x_i)\rvert^2_2 +\lambda \lVert \Lambda \rVert^2_{\mathcal{H}}$$
When $\lambda=0$, all methods are equivalent.

We may want to define a sharpness parameter $\alpha$ to control how sharply one follows the errors. Because as it is now, the $DlogD$ term dampens the errors.
We can define 
$$D=Diag(\exp((1-\alpha)\log([E_i(C)]_{kk})))$$
When $\alpha=0$, we recover the dumb version. When $\alpha=1$, we recover the linearised version.

# Solving for C

the value at the minimum of the linearised version is 

$$\sum_{k=1}^p\lambda_k \log([E_i(C)]_{kk})D_k(D_kK_kD_k+\lambda_k I)^{-1}D_k\log([E_i(C)]_{kk})$$

In the dumb case, we have $\log([E_i(C)]_{kk})=\log [(Y(x_i)-M_k(x_i))^2]+1_{training}2\log(c_k)$ and $D=I$, so we can solve as
$$2\log c_k = -\frac{1_{training}^T(K+\lambda I)^{-1}\log [(Y(x_i)-M_k(x_i))^2]}{1_{training}^T(K+\lambda I)^{-1}1_{training}}