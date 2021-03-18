import scipy.linalg as scpla
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from TeachMyAgent.teachers.utils.torch import get_weights, set_weights, to_float_tensor
import copy


class AbstractDistribution(object):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization
    algorithms in order to perform exploration in parameter space. In
    literature, they are also known as high level policies.

    """

    def sample(self):
        """
        Draw a sample from the distribution.

        Returns:
            A random vector sampled from the distribution.

        """
        raise NotImplementedError

    def log_pdf(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            x (np.ndarray): the point where the log pdf is calculated

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Compute the probability density function in the specified point

        Args:
            x (np.ndarray): the point where the pdf is calculated

        Returns:
            The value of the pdf in the specified point.

        """
        raise np.exp(self.log_pdf(x))


class Distribution(AbstractDistribution):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization
    algorithms in order to perform exploration in parameter space. In
    literature, they are also known as high level policies.

    """

    def mle(self, theta, weights=None):
        """
        Compute the (weighted) maximum likelihood estimate of the points,
        and update the distribution accordingly.

        Args:
            theta (np.ndarray): a set of points, every row is a sample
            weights (np.ndarray, None): a vector of weights. If specified
                                        the weighted maximum likelihood
                                        estimate is computed instead of the
                                        plain maximum likelihood. The number of
                                        elements of this vector must be equal
                                        to the number of rows of the theta
                                        matrix.

        """
        raise NotImplementedError

    def diff_log(self, theta):
        """
        Compute the derivative of the gradient of the probability denstity
        function in the specified point.

        Args:
            theta (np.ndarray): the point where the gradient of the log pdf is calculated

        Returns:
            The gradient of the log pdf in the specified point.

        """
        raise NotImplementedError

    def diff(self, theta):
        """
        Compute the derivative of the probability density function, in the
        specified point. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick, i.e.:

        .. math::
            \\nabla_{\\rho}p(\\theta)=p(\\theta)\\nabla_{\\rho}\\log p(\\theta)

        Args:
            theta (np.ndarray): the point where the gradient of the pdf is
            calculated.

        Returns:
            The gradient of the pdf in the specified point.

        """
        return self(theta) * self.diff_log(theta)

    def get_parameters(self):
        """
        Getter.

        Returns:
             The current distribution parameters.

        """
        raise NotImplementedError

    def set_parameters(self, rho):
        """
        Setter.

        Args:
            rho (np.ndarray): the vector of the new parameters to be used by
                              the distribution

        """
        raise NotImplementedError

    @property
    def parameters_size(self):
        """
        Property.

        Returns:
             The size of the distribution parameters.

        """
        raise NotImplementedError


class TorchDistribution(AbstractDistribution, ABC):
    """
    Interface for a generic PyTorch distribution.
    A PyTorch distribution is a distribution implemented using PyTorch.
    Functions ending with '_t' use tensors as input, and also as output when
    required.

    """

    def __init__(self, use_cuda):
        """
        Constructor.

        Args:
            use_cuda (bool): whether to use cuda or not.

        """
        self._use_cuda = use_cuda

    def entropy(self):
        """
        Compute the entropy of the policy.

        Returns:
            The value of the entropy of the policy.

        """

        return self.entropy_t().detach().cpu().numpy()

    def entropy_t(self):
        """
        Compute the entropy of the policy.

        Returns:
            The tensor value of the entropy of the policy.

        """
        raise NotImplementedError

    def mean(self):
        """
        Compute the mean of the policy.

        Returns:
            The value of the mean of the policy.

        """
        return self.mean_t().detach().cpu().numpy()

    def mean_t(self):
        """
        Compute the mean of the policy.

        Returns:
            The tensor value of the mean of the policy.

        """
        raise NotImplementedError

    def log_pdf(self, x):
        x = to_float_tensor(x, self._use_cuda)
        return self.log_pdf_t(x).detach().cpu().numpy()

    def log_pdf_t(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            x (torch.Tensor): the point where the log pdf is calculated

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the distribution

        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the trainable distribution parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._use_cuda


class GaussianTorchDistribution(TorchDistribution):

    def __init__(self, mu, chol_flat, use_cuda):
        super().__init__(use_cuda)
        self._dim = mu.shape[0]

        self._mu = nn.Parameter(torch.as_tensor(mu, dtype=torch.float32), requires_grad=True)
        self._chol_flat = nn.Parameter(torch.as_tensor(chol_flat, dtype=torch.float32), requires_grad=True)

        self.distribution_t = MultivariateNormal(self._mu, scale_tril=self.to_tril_matrix(self._chol_flat, self._dim))

    def __copy__(self):
        return GaussianTorchDistribution(self._mu, self._chol_flat, self.use_cuda)

    def __deepcopy__(self, memodict=None):
        return GaussianTorchDistribution(copy.deepcopy(self._mu), copy.deepcopy(self._chol_flat), self.use_cuda)

    @staticmethod
    def to_tril_matrix(chol_flat, dim):
        if isinstance(chol_flat, np.ndarray):
            chol = np.zeros((dim, dim))
            exp_fun = np.exp
        else:
            chol = torch.zeros((dim, dim))
            exp_fun = torch.exp

        d1, d2 = np.diag_indices(dim)
        chol[d1, d2] += exp_fun(chol_flat[0: dim])
        ld1, ld2 = np.tril_indices(dim, k=-1)
        chol[ld1, ld2] += chol_flat[dim:]

        return chol

    @staticmethod
    def flatten_matrix(mat, tril=False):
        if not tril:
            mat = scpla.cholesky(mat, lower=True)

        dim = mat.shape[0]
        d1, d2 = np.diag_indices(dim)
        ld1, ld2 = np.tril_indices(dim, k=-1)

        return np.concatenate((np.log(mat[d1, d2]), mat[ld1, ld2]))

    def entropy_t(self):
        return self.distribution_t.entropy()

    def mean_t(self):
        return self.distribution_t.mean

    def log_pdf_t(self, x):
        return self.distribution_t.log_prob(x)

    def sample(self):
        return self.distribution_t.rsample()

    def covariance_matrix(self):
        return self.distribution_t.covariance_matrix.detach().numpy()

    def set_weights(self, weights):
        set_weights([self._mu], weights[0:self._dim], self._use_cuda)
        set_weights([self._chol_flat], weights[self._dim:], self._use_cuda)
        # This is important - otherwise the changes will not be reflected!
        self.distribution_t = MultivariateNormal(self._mu, scale_tril=self.to_tril_matrix(self._chol_flat, self._dim))

    def get_weights(self):
        mu_weights = get_weights([self._mu])
        chol_flat_weights = get_weights([self._chol_flat])

        return np.concatenate([mu_weights, chol_flat_weights])

    def parameters(self):
        return [self._mu, self._chol_flat]
