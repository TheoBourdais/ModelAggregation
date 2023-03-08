import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize


class GPModel:
    def __init__(self, interior_collocation_points, exterior_collocation_points, K, DK, DDK, g, f, tau, grad_tau):
        self.int_points = interior_collocation_points
        self.ext_points = exterior_collocation_points
        self.K = K
        self.DK = DK
        self.DDK = DDK
        self.f = f
        self.g = g
        self.tau = tau
        self.grad_tau = grad_tau
        self._nugget = 1e-10
        self.optimal_parameter = None

    def set_kernel(self, k, dk, ddk):
        del self.K_mat
        self.K = k
        self.DK = dk
        self.DDK = ddk

    @property
    def K_mat(self):
        try:
            return self._kmat
        except:
            self._kmat = self.get_kmat()
            return self._kmat

    @K_mat.deleter
    def K_mat(self):
        try:
            del self._kmat
        except:
            pass
        try:
            del self._eigenvalues
        except:
            pass
        try:
            del self._eigenvectors
        except:
            pass

        try:
            del self._kinv
        except:
            pass
        try:
            del self._z
        except:
            pass
        try:
            del self._norm
        except:
            pass

    @property
    def K_mat_inverse(self):
        try:
            return self._kinv
        except:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.K_mat)
            self._kinv = self._eigenvectors@np.diag(
                [1/eigenval for eigenval in self._eigenvalues])@self._eigenvectors.T
            return self._kinv

    @property
    def norm(self):
        try:
            return self._norm
        except AttributeError:
            self._norm = np.dot(self.z, self.K_mat_inverse@self.z)
            return self._norm

    def loss(self):
        res = self.norm
        for elem in self._eigenvalues:
            if elem < 0:
                return np.Inf
            else:
                res += np.log(elem)
        return res

    @property
    def z(self):
        try:
            return self._z
        except AttributeError as e:
            raise Exception('You must fit the model first')

    @z.setter
    def z(self, val):
        print('be careful when manually setting z')
        self._z = val

    def get_kmat(self):
        all_points = np.concatenate([self.int_points, self.ext_points])
        k1 = GPModel.compute_kernel_matrix(self.K, all_points)
        k2 = GPModel.compute_kernel_matrix(self.DDK, self.int_points)
        k12 = GPModel.compute_kernel_matrix(
            self.DK, all_points, self.int_points)
        return np.block([[k1, k12], [k12.T, k2]])+self._nugget*np.eye(k1.shape[1]+k12.shape[1])

    def compute_kernel_matrix(k, x, y=None):
        if y is None:
            res = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(i, x.shape[0]):
                    res[i, j] = 0.5*(1+int(i != j))*k(x[i], x[j])
            return res+res.T
        return np.array([[k(xi, yi) for yi in y] for xi in x])

    def w_to_z(self, w):
        if not hasattr(self, 'g_vec') or not hasattr(self, 'f_vec'):
            self.g_vec = np.array([self.g(x) for x in self.ext_points])
            self.f_vec = np.array([self.f(x) for x in self.int_points])
        tau_vec = np.array([self.tau(w) for x in self.int_points])
        return np.concatenate([w, self.g_vec, self.f_vec-tau_vec])

    def get_linearisation(self, w):
        if not hasattr(self, 'g_vec') or not hasattr(self, 'f_vec'):
            self.g_vec = np.array([self.g(x) for x in self.ext_points])
            self.f_vec = np.array([self.f(x) for x in self.int_points])
        tau_vec = np.array([self.tau(w) for x in self.int_points])
        C = np.concatenate([np.zeros_like(w), self.g_vec, self.f_vec-tau_vec])
        H = np.block([
            [np.eye(self.int_points.shape[0])],
            [np.zeros((self.ext_points.shape[0], self.int_points.shape[0]))],
            [-np.diag([self.grad_tau(wi) for wi in w])]
        ])
        return H, C

    def solve_linear_w_optim(self, H, C):
        return -np.linalg.inv(H.T@self.K_mat_inverse@H)@H.T@self.K_mat_inverse@C

    def fit(self, alpha_schedule, w_init=None):
        if w_init is None:
            w_init = np.zeros(self.int_points.shape[0])
        w = w_init
        for alpha in alpha_schedule:
            H, C = self.get_linearisation(w)
            w += alpha*self.solve_linear_w_optim(H, C)
        self._z = self.w_to_z(w)

    def kvec(self, x):
        res = []
        res_der = []
        for point in self.int_points:
            res.append(self.K(x, point))
            res_der.append(self.DK(x, point))
        for point in self.ext_points:
            res.append(self.K(x, point))
        return np.array(res+res_der)

    def find_optimal_hyperparameter(self, k, dk, ddk, starting_guess, alpha_schedule, w_init=None):
        save_vals = {}

        def opti_func(param_log):
            param = np.exp(param_log)
            try:
                prev_z = self.z
                prev_k = self.K_mat_inverse
            except:
                prev_k = self.K_mat_inverse
            try:
                return save_vals[tuple(param)]
            except KeyError:
                self.set_kernel(
                    k=lambda x, y: k(x, y, param),
                    dk=lambda x, y: dk(x, y, param),
                    ddk=lambda x, y: ddk(x, y, param)
                )
                self.fit(alpha_schedule, w_init)
                save_vals[tuple(param)] = self.loss()
                # print(f'variance : {self.variance}')
                # print(f'loss : {self.loss()}')
                # try:
                #    print(f'change of z : {np.linalg.norm(self.z-prev_z)}')
                # except:
                #    print(f'first z : {np.linalg.norm(self.z)}')
                # print(f'change of K : {np.linalg.norm(self.K_mat_inverse-prev_k)}')
                return self.loss()
        sigmas = np.logspace(-2, -0.9, 15)

        res = minimize(opti_func, np.log(starting_guess),
                       method='Nelder-Mead', tol=1e-6)
        self.optimal_parameter = np.exp(res.x)
        plt.plot([s for s in sigmas], [opti_func(np.log([param]))
                 for param in sigmas])
        plt.axvline(self.optimal_parameter[0])
        plt.xscale('log')
        plt.yscale('log')
        self.set_kernel(
            k=lambda x, y: k(x, y, self.optimal_parameter),
            dk=lambda x, y: dk(x, y, self.optimal_parameter),
            ddk=lambda x, y: ddk(x, y, self.optimal_parameter)
        )
        self.fit(alpha_schedule, w_init)

    def __call__(self, x):
        return np.dot(self.kvec(x), self.K_mat_inverse@self.z)


class GPInterface:

    def get_kmat(self) -> callable:
        print('get_kmat to be implemented in specific case')

    @property
    def K_mat(self):
        try:
            return self._kmat
        except:
            self._kmat = self.get_kmat()
            return self._kmat

    @K_mat.deleter
    def K_mat(self) -> None:
        try:
            del self._kmat
        except:
            pass
        try:
            del self._eigenvalues
        except:
            pass
        try:
            del self._eigenvectors
        except:
            pass

        try:
            del self._kinv
        except:
            pass
        try:
            del self._z
        except:
            pass
        try:
            del self._norm
        except:
            pass

    @property
    def K_mat_inverse(self) -> np.array:
        try:
            return self._kinv
        except:
            eig, eigv = np.linalg.eigh(self.K_mat)
            inverse = eigv@np.diag(
                [1/eigenval for eigenval in eig])@eigv.T
            self._kinv = inverse
            self._eigenvalues, self._eigenvectors = eig, eigv
            return self._kinv

    @property
    def norm(self) -> float:
        try:
            return self._norm
        except AttributeError:
            self._norm = np.dot(self.z, self.K_mat_inverse@self.z)
            return self._norm

    @property
    def loss(self) -> float:
        res = self.norm
        for elem in self._eigenvalues:
            if elem < 0:
                return np.Inf
            else:
                res += np.log(elem)
        return res

    @property
    def z(self):
        try:
            return self._z
        except AttributeError as e:
            raise Exception('You must fit the model first')

    @z.setter
    def z(self, val):
        print('be careful when manually setting z')
        self._z = val

    def kvec(self, x: np.array) -> np.array:
        print('kvec to be implemented in specific application')

    def covariate(self, x):
        KV = self.kvec(x)
        return np.dot(KV, self.K_mat_inverse@KV)

    def variance(self, x):
        return self.K(x, x)-self.covariate(x)

    def __call__(self, x):
        return np.dot(self.kvec(x), self.K_mat_inverse@self.z)

    def __hash__(self):
        return hash(self.K_mat_inverse.tostring())


class GPAggregate():

    def __init__(self, models):
        self.models = models
        self.K = models[0].K
        self.setup_cov_dict()
        self._nugget = 1e-10

    def setup_cov_dict(self):
        self._model_covs = {}
        for model in self.models:
            try:
                self._model_covs.update(model._model_covs)
                model._model_covs = self._model_covs
            except:
                pass

    def update_cov_dict(model1, model2, cov_dict):
        if model1 == model2:
            cov_dict[(model1, model2)] = model1.K_mat_inverse
        else:
            cov_dict[(model1, model2)] = model1.K_mat_inverse@model1.mat_covariance(
                model2)@model2.K_mat_inverse
            cov_dict[(model2, model1)] = cov_dict[(model1, model2)].T

    def cov_matrix(self, x):
        res = np.zeros((len(self.models), len(self.models)))
        for i in range(res.shape[0]):
            for j in range(i, res.shape[0]):
                res[i, j] = GPAggregate.covariate_models(
                    self.models[i], self.models[j], x, self._model_covs)
                res[j, i] = res[i, j]
        return res+self._nugget*np.eye(len(self.models))

    def K_mat_inverse(self, x):
        return np.linalg.inv(self.cov_matrix(x))

    def kvec(self, x):
        return np.array([model.covariate(x) for model in self.models])

    def covariate(self, x):
        v = self.kvec(x)
        return np.dot(v, self.K_mat_inverse(x)@v)

    def covariate_models(model1, model2, x, dict_used):
        if hasattr(model1, 'models'):
            return np.dot(model1.K_mat_inverse(x)@model1.kvec(x), np.array([GPAggregate.covariate_models(model, model2, x, dict_used) for model in model1.models]))

        elif hasattr(model2, 'models'):
            return np.dot(model2.K_mat_inverse(x)@model2.kvec(x), np.array([GPAggregate.covariate_models(model, model1, x, dict_used) for model in model2.models]))

        else:
            try:
                return np.dot(model1.kvec(x), dict_used[(model1, model2)]@model2.kvec(x))
            except KeyError:
                GPAggregate.update_cov_dict(model1, model2, dict_used)
                return np.dot(model1.kvec(x), dict_used[(model1, model2)]@model2.kvec(x))

    def variance(self, x):
        return self.K(x, x)-self.covariate(x)

    def __call__(self, x):
        return np.dot(self.kvec(x), self.K_mat_inverse(x)@np.array([model(x) for model in self.models]))

    def __hash__(self):
        return hash(tuple(self.models))


class pdeGP(GPInterface):
    def __init__(self, interior_collocation_points: np.array, exterior_collocation_points: np.array, K: callable, DK: callable, DDK: callable, g: callable, f: callable, tau: callable, grad_tau: callable):
        self.int_points = interior_collocation_points
        self.ext_points = exterior_collocation_points
        self.K = K
        self.DK = DK
        self.DDK = DDK
        self.f = f
        self.g = g
        self.tau = tau
        self.grad_tau = grad_tau
        self._nugget = 1e-10
        self.optimal_parameter = None
        self.is_aggregate = False

    @classmethod
    def from_model(cls, other_model):
        return cls(interior_collocation_points=None, exterior_collocation_points=None, K=other_model.K, DK=other_model.DK, DDK=other_model.DDK, g=other_model.g, f=other_model.f, tau=other_model.tau, grad_tau=other_model.grad_tau)


    def set_kernel(self, k, dk, ddk):
        del self.K_mat
        self.K = k
        self.DK = dk
        self.DDK = ddk

    def get_kmat(self):
        all_points = np.concatenate([self.int_points, self.ext_points])
        k1 = pdeGP.compute_kernel_matrix(self.K, all_points)
        k2 = pdeGP.compute_kernel_matrix(self.DDK, self.int_points)
        k12 = pdeGP.compute_kernel_matrix(self.DK, all_points, self.int_points)
        return np.block([[k1, k12], [k12.T, k2]])+self._nugget*np.eye(k1.shape[1]+k12.shape[1])

    def compute_kernel_matrix(k, x, y=None):
        if y is None:
            res = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(i, x.shape[0]):
                    res[i, j] = 0.5*(1+int(i != j))*k(x[i], x[j])
            return res+res.T
        return np.array([[k(xi, yi) for yi in y] for xi in x])
    
    def mat_covariance(self,other_model):
        all_points1=np.concatenate([self.int_points, self.ext_points])
        all_points2=np.concatenate([other_model.int_points, other_model.ext_points])
        k11=pdeGP.compute_kernel_matrix(self.K, all_points1,all_points2)
        k12=pdeGP.compute_kernel_matrix(self.DK, all_points1, other_model.int_points)
        k21=pdeGP.compute_kernel_matrix(self.DK, self.int_points, all_points2)
        k22=pdeGP.compute_kernel_matrix(self.DDK, self.int_points,other_model.int_points)
        return np.block([[k11, k12], [k21, k22]])

    def w_to_z(self, w):
        if not hasattr(self, 'g_vec') or not hasattr(self, 'f_vec'):
            self.g_vec = np.array([self.g(x) for x in self.ext_points])
            self.f_vec = np.array([self.f(x) for x in self.int_points])
        tau_vec = np.array([self.tau(w) for x in self.int_points])
        return np.concatenate([w, self.g_vec, self.f_vec-tau_vec])

    def solve_linearisation(self, w):
        if not hasattr(self, 'g_vec') or not hasattr(self, 'f_vec'):
            self.g_vec = np.array([self.g(x) for x in self.ext_points])
            self.f_vec = np.array([self.f(x) for x in self.int_points])
        tau_vec = np.array([self.tau(w) for x in self.int_points])
        C = np.concatenate([np.zeros_like(w), self.g_vec, self.f_vec-tau_vec])
        H = np.block([
            [np.eye(self.int_points.shape[0])],
            [np.zeros((self.ext_points.shape[0], self.int_points.shape[0]))],
            [-np.diag([self.grad_tau(wi) for wi in w])]
        ])
        return -np.linalg.inv(H.T@self.K_mat_inverse@H)@H.T@self.K_mat_inverse@C

    def kvec(self, x):
        res = []
        res_der = []
        for point in self.int_points:
            res.append(self.K(x, point))
            res_der.append(self.DK(x, point))
        for point in self.ext_points:
            res.append(self.K(x, point))
        return np.array(res+res_der)

    def fit(self, alpha_schedule, w_init=None):
        if w_init is None:
            w_init = np.zeros(self.int_points.shape[0])
        w = w_init
        for alpha in alpha_schedule:
            w += alpha*self.solve_linearisation(w)
        found_z = self.w_to_z(w)
        self._z = found_z
