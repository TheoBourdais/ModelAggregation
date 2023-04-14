import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm import tqdm




class PDESolver():
    def __init__(self,X_int,X_boundary,sigma,name=None) -> None:
        self.X_int=X_int
        self.X_boundary=X_boundary
        self.X_all=np.concatenate([X_int,X_boundary])
        self.Nd=X_int.shape[0]
        self.sigma=sigma
        self.memory={}
        if name is not None:
            self.name=name

    
    def fit(self,f,g,tau,dtau,nugget=1e-5):
        self.nugget=nugget
        self.K_mat = PDESolver.get_kernel_matrix(self.X_all,self.Nd,self.sigma,nugget)
        self.L=np.linalg.inv(np.linalg.cholesky(self.K_mat))
        self.a=PDESolver.gauss_newton(self.X_int,self.X_boundary,self.L,f,g,tau,dtau)
    

    def covariate_with_other(self,other_GP,x,sigma):
        M=self.find_covariance_matrix(other_GP,sigma)
        v1=self.L.T@self.L@PDESolver.get_kernel_vector(self.X_all,self.Nd,self.sigma,x)
        v2=other_GP.L.T@other_GP.L@PDESolver.get_kernel_vector(other_GP.X_all,other_GP.Nd,other_GP.sigma,x)
        return np.einsum('ij,ij->j',v1,M@v2)
    
    def covariate_with_sol(self,x,sigma=None):
        if sigma is None:
            v= PDESolver.get_kernel_vector(self.X_all,self.Nd,self.sigma,x)
            return np.linalg.norm(self.L@v,axis=0)**2
        return self.covariate_with_other(self,x,sigma)
    
    def find_covariance_matrix(self,other_GP,sigma):
        try:
            return self.memory[other_GP][sigma]
        except:
            M=PDESolver.get_covariance_matrix(self.X_all,other_GP.X_all,self.Nd,other_GP.Nd,sigma,self.nugget)
            try:
                self.memory[other_GP][sigma]=M
            except:
                self.memory[other_GP]={sigma:M}
            return M
    

    def __call__(self,x):
        return np.dot(self.L.T@self.L@self.a,PDESolver.get_kernel_vector(self.X_all,self.Nd,self.sigma,x))


    def laplacian(self,x):
        return np.dot(self.L.T@self.L@self.a,PDESolver.get_laplacian_kernel_vector(self.X_all,self.Nd,self.sigma,x))
    
    def __repr__(self) -> str:
        try:
            return self.name
        except:
            return 'PDE Solver with K_mat : '+self.K_mat.__repr__()







    def make_alpha(x_int,x_ext,f,g,tau):
        alpha=lambda z,g,f,t:np.concatenate([z,g,f-t(z)])
        g_vec=np.array([g(x) for x in x_ext])
        f_vec=np.array([f(x) for x in x_int])
        return partial(alpha,g=g_vec,f=f_vec,t=tau)
    
    def differential_matrix(z,N_int,N_ext,dtau):
        return np.block([
            [-np.eye(N_int)],
            [np.zeros((N_ext,N_int))],
            [np.diag(dtau(z))]
        ])
    
    def gauss_newton(x_int,x_ext,L,f,g,tau,dtau):
        z=np.zeros(x_int.shape[0])
        alpha=PDESolver.make_alpha(x_int,x_ext,f,g,tau)
        dz=3*np.ones_like(z)
        progress = tqdm()
        while np.linalg.norm(dz,np.inf)>1e-6:
            H=PDESolver.differential_matrix(z,x_int.shape[0],x_ext.shape[0],dtau)
            target=L@alpha(z)
            mat=L@H
            dz=np.linalg.lstsq(mat,target,rcond=None)[0]
            z+=dz
            progress.set_description(f'Current residual {np.linalg.norm(dz,np.inf):.3e}')
            progress.update()
        progress.close()
        return alpha(z)
    


    def get_kernel_matrix(X,Nd,sigma,nugget):
        return PDESolver.get_covariance_matrix(X,X,Nd,Nd,sigma,nugget)
        distances=pairwise_distances(X)**2
        K11=np.exp(-distances/2/sigma**2)
        K12=-K11[:,:Nd]*(distances[:,:Nd]-2*sigma**2)/sigma**4
        K22=K11[:Nd,:Nd]*((distances[:Nd,:Nd]-4*sigma**2)**2-8*sigma**4)/sigma**8
        return np.block([[K11,K12],[K12.T,K22]])
    
    def get_covariance_matrix(X,Y,Nd_x,Nd_y,sigma,nugget):
        distances=pairwise_distances(X,Y=Y)**2
        nugget_mat=nugget*(distances == 0).astype(int)
        K11=np.exp(-distances/2/sigma**2)
        K12=-K11[:,:Nd_y]*(distances[:,:Nd_y]-2*sigma**2)/sigma**4
        K21=-K11[:Nd_x,:]*(distances[:Nd_x,:]-2*sigma**2)/sigma**4
        K22=K11[:Nd_x,:Nd_y]*((distances[:Nd_x,:Nd_y]-4*sigma**2)**2-8*sigma**4)/sigma**8
        K11+=nugget_mat
        K22+=nugget_mat[:Nd_x,:Nd_y]
        return np.block([[K11,K12],[K21,K22]])
    
    def get_kernel_vector(X,Nd,sigma,x):
        distances=pairwise_distances(X,Y=x)**2
        K1=np.exp(-distances/2/sigma**2)
        K2=-K1[:Nd]*(distances[:Nd]-2*sigma**2)/sigma**4
        return np.concatenate([K1,K2])
    
    def get_laplacian_kernel_vector(x_all,Nd,sigma,x):
        distances=pairwise_distances(x_all,Y=x)**2
        K0=np.exp(-distances/2/sigma**2)
        K1=K0*(distances-2*sigma**2)/sigma**4
        K2=-K0[:Nd]*((distances[:Nd]-4*sigma**2)**2-8*sigma**4)/sigma**8
        return np.concatenate([K1,K2])
    








class Aggregate():
    def __init__(self,models,sigma,nugget=1e-5) -> None:
        self.models=np.array(models)
        self.nugget=nugget
        self.sigma=sigma
    
    def covariate_inner_models_with_sol(self,x,sigma):
        return np.array(list(map(lambda model:model.covariate_with_sol(x,sigma),self.models))).T
    
    def inner_models_cov_matrix(self,x):
        triangular_indices=np.tril_indices(self.models.shape[0])
        pairs=np.stack([self.models[triangular_indices[0]],self.models[triangular_indices[1]]],axis=-1)
        covs=np.array(list(map(lambda model_pair:Aggregate.covariate_models(model_pair[0],model_pair[1],x,self.sigma),pairs)))
        cov_mat=np.zeros((self.models.shape[0],self.models.shape[0],x.shape[0]))
        cov_mat[triangular_indices]=covs
        cov_mat=cov_mat+np.transpose(cov_mat, (1, 0, 2))
        indexes=np.arange(self.models.shape[0])
        cov_mat[indexes,indexes,:]-=np.diagonal(cov_mat,axis1=0, axis2=1).T/2-self.nugget
        return cov_mat.T
    
    def covariate_with_sol(self,x,sigma):
        return np.dot(self.alpha(x),self.covariate_inner_models_with_sol(x,sigma))
    
    def alpha(self,x):

        #return np.linalg.solve(self.inner_models_cov_matrix(x),self.covariate_inner_models_with_sol(x,self.sigma))
        COV_mat=self.inner_models_cov_matrix(x)
        COV_Y=self.covariate_inner_models_with_sol(x,self.sigma)
        print('COV mat',COV_mat)
        print('COV Y',COV_Y)
        alphas=np.stack(list(map(lambda A,B:np.linalg.lstsq(A,B,rcond=None)[0],COV_mat,COV_Y)),axis=0)
        return alphas


    
    def __call__(self, x):
        M=np.array(list(map(lambda model:model(x),self.models))).T
        print('M',M)
        alpha=self.alpha(x)
        print('alpha',alpha)
        return np.einsum('ij,ij->i',alpha,M)
    

    def covariate_models(model1,model2,x,sigma):
        if hasattr(model1,'models'):
            covs=np.array(list(map(lambda model:Aggregate.covariate_models(model,model2,x,sigma),model1.models)))
            return np.dot(model1.alpha(x),covs)
        if hasattr(model2,'models'):
            covs=np.array(list(map(lambda model:Aggregate.covariate_models(model,model1,x,sigma),model2.models)))
            return np.dot(model2.alpha(x),covs)
        return model1.covariate_with_other(model2,x,sigma)