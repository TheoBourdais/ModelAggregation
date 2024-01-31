# Project Ideas

## Link to neural network

There are two ways of rationalising what neural networks do. 
* have a kernel that uses the feature map of one NN
* attention layers is a model aggregation -> put the model inside of the kernel

## Solving the aggregation

One can use https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html to optimize


## Covariance

If we suppose $M$ and $\alpha$ to be independant, then 
$$Cov(\alpha^TM)=Tr(Cov(\alpha)Cov(M))+\mathbb{E}[M]^TCov(\alpha) \mathbb{E}[M] +\mathbb{E}[\alpha]^TCov(M) \mathbb{E}[\alpha] $$

We have two possible formulations: 
### loss minimization
$$\alpha=(k(\cdot,x_1),\dots,k(\cdot,x_n))V$$ 
with $V\in\mathbb{R}^{n\times d}$ $d$ is the number of models, $n$ is the number of points.
and $$\alpha=\argmin_{\alpha\in\mathcal{H}_\kappa} \sum_{k=1}^N \left(Y(x_i)-\alpha(x_i)^TM(x_i)\right)^2$$

So for $$ \mathcal{M}=
                Diag(M(x_i)^T)K=Diag(M(x_i)^T)\begin{pmatrix} 
            \kappa(x_1,z_1) & \kappa(x_1,z_2) & \dots \\
            \vdots & \ddots & \\
            \kappa(x_N,z_1) &        & \kappa(x_N,z_k) 
            \end{pmatrix}$$
We get 
$$V=\argmin_{v\in\mathbb{R}^{kn}} \left\lVert Y-\mathcal{M}v \right\rVert^2+\lambda \lVert v \rVert^2$$

Then, 
$$V=\left(\mathcal{M}^T\mathcal{M}+\lambda \right)^{-1}\mathcal{M}^TY=(KDiag(M(x_i))Diag(M(x_i)^T)K+\lambda)^{-1}K\ Diag(M(x_i))Y$$
and 
$$V=\mathcal{M}^T\left(\mathcal{M}\mathcal{M}^T+\lambda \right)^{-1}Y=K\ Diag(M(x_i))(Diag(M(x_i))K^2Diag(M(x_i)^T)+\lambda)^{-1}Y$$

And our final prediction is 
$$\alpha(x)=\kappa(x,X)K\ Diag(M(x_i))(Diag(M(x_i))K^2Diag(M(x_i)^T)+\lambda)^{-1}Y$$

### Conditional expecation

Write $W=(\xi(x),\xi(X),Z)$, we have 
$$W\sim\mathcal{N}(0,\begin{pmatrix} 
            \kappa(x,x) & \kappa(x,X) & 0 \\
            \kappa(X,x) & \kappa(X,X) & 0 \\
            0 & 0 & \lambda I
            \end{pmatrix})$$
Write $K_W$ the covariance matrix of $W$, $K=\kappa(X,X)$. Then, 

$$\phi= \begin{pmatrix} 
            0 & Diag(M(x_i)^T) & I
            \end{pmatrix}$$
The condition $\xi(X_i)^TM(X_i)+Z_i$ can be written as $(\phi W)_i$, so 
$$W\lvert\xi(X_i)^TM(X_i)+Z_i=W\lvert\phi W\sim\mathcal{N}(K_W\phi^T(\phi K_W\phi^T)^{-1}\phi W,K_W-K_W\phi^T(\phi K_W\phi^T)^{-1}\phi K_W)$$
Thus, 
$$\xi(x)\lvert\xi(X_i)^TM(X_i)+Z_i=Y_i\sim\mathcal{N}(\tilde{\alpha}(x),\sigma^2(x))$$
with
$$\tilde{\alpha}(x)=\kappa(x,X)Diag(M(x_i))(Diag(M(x_i)^T)K\ Diag(M(x_i))+\lambda I)^{-1}Y$$
$$\sigma^2(x)=\kappa(x,x)-\kappa(x,X)Diag(M(x_i))(Diag(M(x_i)^T)K\ Diag(M(x_i))+\lambda I)^{-1}Diag(M(x_i)^T)\kappa(X,x)$$

### Compound kernel

If $M$ is a function (not random), we can define a compound kernel as
$$\Gamma(x,y)=M(x)^T\kappa(x,y)M(y)$$
Then, for $\zeta\sim\mathcal{N}(0,\Gamma)$, we notice that 
$$M(x)^T\tilde{\alpha}(x)=\mathbb{E}[\zeta(x)\lvert\zeta(X_i)+Z_i=Y_i]$$
And
$$\mathbb{V}ar[\zeta(x)\lvert\zeta(X_i)+Z_i=Y_i]=M(x)^T\sigma^2(x)M(x)=\mathbb{V}ar[M(x)^T\xi(x)\lvert\xi(X_i)^TM(X_i)+Z_i=Y_i]$$


## Observation on the aggregation

The Gaussian process doesn't know that the models are supposed to regress $Y$. Instead, it uses them as features and tries to do the regression itself. But if we give few data, or our aggregator is weak, then we will get poor regression. See example in the [notebook](/agregator/pathological_case.ipynb). Example image:

![pathological case](/images/pathological_case.png)
![pathological case](/images/pathological_case_gaussian.png)

The solution might be to change the regularisation, by adding the fact that aggregator should avoid using bad models. i.e. adding the term
$$\lambda_2\sum_{j=1}^{k}\sum_{i=1}^N \alpha_j^2(x_i)(M_j(x_i)-Y_i)^2$$

We are now solving, for $E_i^2=(Diag(M_k(x_i))-Y_i)^2$
$$V=\argmin_{v\in\mathbb{R}^{kn}} \left\lVert Y-\mathcal{M}v \right\rVert^2+\lambda \lVert v \rVert^2+\lambda_2 v^TKDiag(E_i^2)Kv$$
So 
$$V=\left(\mathcal{M}^T\mathcal{M}+\lambda I+\lambda_2 KDiag(E_i^2)K\right)^{-1}\mathcal{M}^TY$$
Using the definition of $\mathcal{M}=Diag(M(x_i)^T)K$, we get 
$$V=\left(K[Diag(M(x_i))Diag(M(x_i)^T)+\lambda_2 Diag(E_i^2)]K+\lambda I\right)^{-1}K\ Diag(M(x_i))Y$$

> $$(P^{-1}+B^TR^{-1}B)^{-1}B^TR^{-1}=PB^T(BPB^T+R)^{-1}$$

Since before we were forming the matrix $\mathcal{M}$ and solving a regularized least square, we can make this matrix larger and define

$$\tilde{\mathcal{M}}=\begin{pmatrix} 
            \mathcal{M} \\
            \sqrt{\lambda_2}Diag(E_i)K
            \end{pmatrix}$$

This seem to be like giving models a covariances 
$$Cov(M(x_i))=\begin{pmatrix} 
            (M_1(x_i)-Y_i)^2 & \dots & 0 \\
            \vdots & \ddots & \\
            0 &        & (M_k(x_i)-Y_i)^2 
            \end{pmatrix}$$

And then computing 
$$\alpha(x_i)^TCov(M(x_i))\alpha(x_i)$$

Which is one part of the covariance of $\alpha(x_i)^TM(x_i)$. Since the classical regularization is 
$$\lVert f \rVert_K^2=Cov(\langle f,\xi \rangle)$$
Maybe we could modify this covariance computation to take into account the covariance of the models.


>Idead: two stage version. One stage models the covariance of the models, the other stage uses this covariance to do the regression.

> To do:
> - create a measure of how the errors are distributed, i.e. if the errors are spatially correlated and if some models are better at some spots than others
> - Such a measure could simply be comparing 
> $$\min_{j} \frac{1}{N}\sum_{i=1}^N (M_j(x_i)-Y_i)^2$$
> with
> $$\frac{1}{N}\sum_{i=1}^N \min_{j}(M(x_i)-Y_i)^2$$
> - Result on boston dataset shows best possible could divide the error by 2
> - Create the attention kernel
> - create a difference between training and validation data in the aggregation
>   - i.e. model overfitting


## Attention kernel

Chose a kernel $k$, then you can define
$$K_{i,j}(x,y)=k(M_i(x),M_j(y))$$
