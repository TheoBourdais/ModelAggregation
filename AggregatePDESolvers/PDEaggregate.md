We have two proposed methodologies for aggregating the solvers. Let's suppose we found a basis $\phi$ by performing PCA on $Y$. 
Then we can have the following formula:
$$M_A(f)=\sum_{i=1}^{n} \phi_i \alpha(f)^T\left(\begin{matrix}\langle \phi_i ,M_1(f)\rangle\\\dots\\\langle \phi_i ,M_p(f)\rangle\end{matrix}\right)$$


Another approach is pointwise, where we have the following formula:
$$M_A(f)=\left(\sum_{i=1}^{n} \phi_i \alpha_i(f)^T\right)\left(\begin{matrix}M_1(f)\\ \vdots\\M_p(f)\end{matrix}\right)$$

or 
$$M_A(f)=\left(\sum_{i=1}^{n} \phi_i \alpha_i(f)^T\right)M(f)$$

if no interraction betweeen the $\alpha_i$, then $\alpha_i(f)=K(f,F)V_i$
So 
$$M_A(f)=\sum_{i=1}^{n} \phi_i * (V_iK(F,f)M(f))$$

Here, 
- $\phi_i$ is a grid $(N_{grid},N_{grid})$, 
- $M(f)$ is $(p,N_{grid},N_{grid})$, 
- $V_i$ is $(N_{samples}\times p)$, 
- $K(F,f)$ is $(N_{samples}\times p,p)$ 
