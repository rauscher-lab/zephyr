import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn as nn

import config
from utils import log1mexp


def init_noise_schedule(T: int = None, Type: str = None, params: dict = None):
    T = config.n_diff_steps if T is None else T
    Type = config.noise_sch_type if Type is None else Type
    params = config.noise_sch_params if params is None else params
    if params is None:
        params = dict() if config.noise_sch_params is None else config.noise_sch_params

    if Type == 'cosine':
        noise_sch = CosineDDPMSch(T=T, **params)
    elif Type == 'cosine2':
        noise_sch = Cosine2Sch(T=T, **params)
    elif Type == 'polynomial':
        noise_sch = PolySch(T=T, **params)
    elif Type == 'polynomial2':
        noise_sch = Poly2Sch(T=T, **params)
    elif Type == 'sigmoid':
        noise_sch = SigmoidSch(T=T, **params)
    elif Type == 'logSNR-linear':
        noise_sch = LogSNRLinearSch(T=T, **params)
    elif Type == 'sigma-linear':
        noise_sch = SigmaLinearSch(T=T, **params)
    elif Type == 'laplace':
        noise_sch = LaplaceSch(T=T, **params)
    else:
        raise ValueError(f"Type {Type} is not supported.")

    return noise_sch


def convert_SNR_range(SNR_min=None, SNR_max=None, alpha_min=None, sigma_min=None):
    if SNR_min is not None and SNR_max is not None:
        assert SNR_min > 0 and SNR_max > 0, f"SNR_min and SNR_max must be positive. Received SNR_min{SNR_min}, SNR_max={SNR_max}"
        alpha_min = np.sqrt(SNR_min / (1 + SNR_min))
        sigma_min = 1 / np.sqrt(1 + SNR_max)
        return alpha_min, sigma_min
    if alpha_min is not None and sigma_min is not None:
        alpha_min, sigma_min = torch.as_tensor(alpha_min).double(), torch.as_tensor(sigma_min).double()
        SNR_min = alpha_min.square() / (1 - alpha_min.square())
        SNR_max = (1 - sigma_min.square()) / sigma_min.square()
        return SNR_min, SNR_max
    raise ValueError(f"Not enough information to convert")


class NoiseSchedule(nn.Module):
    """
    Class that encapsulates attributes and timeseries related to the noise schedule used in diffusion models.
    The noise schedule is defined through the functions alpha(t) and sigma(t), which determine marginal likelihoods of
    intermediate variables x_t ~ N(x_t; alpha(t) x_0, sigma(t) I) where N is a multivariate normal distribution
    alpha(0)~=1 and sigma(0)~=0.

    The noise schedule can be switched to be variance-preserving or variance-exploding, defined by the 'mode' attribute.
    The two modes are linked by matching logSNR(t).
    """

    def __init__(self, T: int = config.n_diff_steps,
                 alpha_min: float | torch.Tensor = None, sigma_min: float | torch.Tensor = None,
                 SNR_min: float = None, SNR_max: float = None, mode='VP', name='', **kwargs):
        """
        Initializes parameters of the noise schedule. Two initialization methods are possible:
        1) Pass values for alpha_min and sigma_min, which defines the minimum values of alpha(t) and sigma(t) in the variance-preserving case.
        2) Pass values for SNR_min and SNR_max, which defines the minimum and maximum values of the signal-to-noise ration (alpha^2/sigma^2)
        Args:
            T: Number of steps used to discretize the noise schedule
            alpha_min: minimum value of alpha(t) used in the variance-preserving mode
            sigma_min: minimum value of sigma(t) used in the variance-preserving mode
            SNR_min: minimum value of the signal-to-noise ratio (alpha^2/sigma^2)
            SNR_max: maximum value of the signal-to-noise ratio (alpha^2/sigma^2)
            mode: mode of the noise schedule. Choices are ['VP','VE'] where 'VP' is variance-preserving and 'VE' is variance-exploding.
            name: name of the noise schedule
            kwargs: extra parameters whose name are saved for repr
        """
        super().__init__()
        if SNR_min is not None and SNR_max is not None:
            alpha_min, sigma_min = convert_SNR_range(SNR_min=SNR_min, SNR_max=SNR_max)
        elif SNR_min != SNR_max:
            raise ValueError(f"Both SNR_min and SNR_max must be given to define the noise schedule")
        if alpha_min is not None and not isinstance(alpha_min, torch.Tensor):
            alpha_min = torch.tensor(alpha_min)
        if sigma_min is not None and not isinstance(sigma_min, torch.Tensor):
            sigma_min = torch.tensor(sigma_min)
        if alpha_min <= 0:
            raise ValueError(f"alpha_min must be strictly positive. A value of {alpha_min} was defined.")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be strictly positive. A value of {sigma_min} was defined.")

        self.T = T
        self.alpha_min = nn.parameter.Buffer(alpha_min.double(), persistent=True)
        self.sigma_min = nn.parameter.Buffer(sigma_min.double(), persistent=True)
        self.mode = mode
        self.name = name
        self.extra_params_name = list(kwargs)

    def alpha(self, t: torch.Tensor):
        if self.mode == 'VP':
            alpha = torch.sigmoid(self.SNR(t, log=True)).sqrt()
        elif self.mode == 'VE':
            alpha = torch.ones_like(t)
        else:
            raise NotImplementedError(f"alpha(t) is not implemented for mode={self.mode!r}")
        return alpha

    def sigma(self, t: torch.Tensor):
        if self.mode == 'VP':
            sigma = torch.sigmoid(-self.SNR(t, log=True)).sqrt()
        elif self.mode == 'VE':
            # In a variance-exploding scheme, alpha(t)=1 -> SNR = 1/sigma^2 -> sigma = 1/sqrt(SNR) = exp(-0.5*log(SNR))
            sigma = torch.exp(-0.5 * self.SNR(t, log=True))
        else:
            raise NotImplementedError(f"sigma(t) is not implemented for mode={self.mode!r}")
        return sigma

    def attr_der(self, t: torch.Tensor, attr: str, **kwargs):
        """
        Calculates the derivative of the given attribute. For example, attr='sigma' calculates dsigma(t)/dt
        Args:
            t: temporal parameter of the schedule in range [0,1]
            attr: name of the method defining the function of time. The method must take t as first input.
            kwargs: extra kwargs passed to the given method

        Returns:
            torch.Tensor with the same shape as input t
        """
        attr_func = getattr(self, attr)
        with torch.enable_grad():
            t_requires_grad = t.requires_grad
            t.requires_grad = True
            der = torch.autograd.grad(attr_func(t, **kwargs).sum(), t)[0].detach()
            t.requires_grad = t_requires_grad

        return der

    def SNR(self, t: torch.Tensor, log=False):
        """
        Calculates the signal-to-noise ratio (alpha^2/sigma^2) for each given times in the interval [0,1]
        Args:
            t: temporal parameter of the schedule in range [0,1]
            log: returns the logged SNR instead

        Returns:
            torch.Tensor with the same shape as input t
        """
        raise NotImplementedError

    @staticmethod
    def SNR_alpha(alpha: torch.Tensor, log=False):
        """
        Calculates the signal-to-noise ratio (alpha^2/(sigma^2) for given values of alpha(t) in the VP case
        Args:
            alpha: values of alpha(t) in range [0,1]
            log: returns the logged SNR instead

        Returns:
            torch.Tensor with the same shape as input t
        """
        SNR_log = 2 * alpha.log() - torch.log1p(-alpha.square())
        if log:
            return SNR_log
        else:
            return SNR_log.exp()

    def logSNR_pdf(self, t: torch.Tensor):
        """
        Calculates the pdf of logSNR(t) at each input noise level t assuming t is uniformly distributed in [0,1]
        Args:
            t: temporal parameter of the schedule in range [0,1]
        Returns:
            torch.Tensor with the same shape as input t
        """
        # Let lambda(t) = logSNR(t) and invert this relationship to view t as a function of lambda i.e. t(lambda)
        # Then, p(lambda) = p(t(lambda)) |d t(lambda)/dlambda| = - d t(lambda)/dlambda (t is uniform and decreases when lambda increases)
        # Furthermore, d t(lambda)/dlambda = 1/(dlambda/d t(lambda))
        dlogSNR_dt = self.attr_der(t, attr='SNR', log=True)
        return -1 / dlogSNR_dt

    def h(self, t: torch.Tensor):
        """
        Returns the scale of the drift in the SDE associated with the noise schedule, dx = h(t) x dt + g(t)dw_t
        Args:
            t: temporal parameter of the schedule in range [0,1]

        Returns:
            torch.Tensor with the same shape as input t
        """
        alpha_der = self.attr_der(t, attr='SNR', log=True)
        alpha = self.alpha(t)
        h = alpha_der / alpha
        return h

    def g(self, t: torch.Tensor):
        """
        Returns the scale of the Wiener process in the SDE associated with the noise schedule, dx = h(t) x dt + g(t)dw_t
        Args:
            t: temporal parameter of the schedule in range [0,1]

        Returns:
            torch.Tensor with the same shape as input t
        """
        dlogSNR_dt = self.attr_der(t, attr='SNR', log=True)
        g = torch.sqrt(-self.sigma(t).square() * dlogSNR_dt)
        return g

    def Karras_t(self, t: torch.Tensor):
        """
        Returns the time variable used in Karras' EDM model, which corresponds to sigma(t) in a var. exploding scheme.
        Args:
            t: temporal parameter of the schedule in range [0,1]

        Returns:
            torch.Tensor with the same shape as input t
        """
        # In Karras' convention, t=0 is noisefull so we need to evaluate at 1-t.
        SNR_log = self.SNR(1 - t, log=True)
        # Karras uses a variance-exploding scheme where alpha(t)=1, sigma(t) = t -> SNR = 1/t^2 -> t = 1/sqrt(SNR)
        Karras_t = torch.exp(-0.5 * SNR_log)  # Karras_t = 1/sqrt(SNR) = exp(-0.5*log(SNR))
        return Karras_t

    def noise_std(self, t: torch.Tensor):
        """
        Calculates the standard deviation of the added noise in the diffusion SDE associated with the noise schedule.
        Args:
            t: temporal parameter of the schedule in range [0,1]

        Returns:
            torch.Tensor with the same shape as input t
        """
        wiener_std = t.diff().abs().sqrt()
        noise_std = self.g(t[:-1]) * wiener_std
        return noise_std

    @property
    def t_arr(self):
        """
        Linearly samples T+1 values of the temporal time parameter in the range [0,1] (inclusive).
        Returns:
            torch.Tensor with shape=(self.T+1,)
        """
        return torch.linspace(0, 1, self.T + 1).double()

    def forward(self, t: torch.Tensor):
        return self.alpha(t)

    def plot_attr_dist(self, attr_name='logSNR', n=100000):
        """
        Plots distribution of a given attribute.
        Args:
            attr_name: name of the method used to calculate the attribute
            n: number of samples used to build the distribution

        Returns:
            None
        """
        t_samples = torch.rand(n)
        if attr_name == 'logSNR':
            attr_samples = self.SNR(t_samples, log=True)
        elif attr_name == 'Karras_t':
            attr_samples = self.Karras_t(t_samples)
        else:
            raise NotImplementedError(f"pdf for attribute {attr_name} is not supported.")
        bins_val, bins_edges = np.histogram(attr_samples, bins=200, density=True)
        bins_centers = (bins_edges[0:-1] + bins_edges[1:]) / 2
        hist_h = plt.plot(bins_centers, bins_val, linestyle='--', label=f'{self}')[0]

        # Also plot the theoretical pdf
        t = torch.linspace(0, 1, 1000)
        if attr_name == 'logSNR':
            pdf_x = self.SNR(t, log=True)
            pdf_y = self.logSNR_pdf(t)
        else:
            raise NotImplementedError(f"pdf for attribute {attr_name} is not supported.")
        plt.plot(pdf_x, pdf_y, label=None, color=hist_h.get_color())
        plt.xlabel(attr_name)
        plt.legend()

    def plot_attr_timeseries(self, attr_name, n: int | None = None):
        """
        Plots timeseries of various attributes that depend on the temporal parameter t in [0,1]
        Args:
            attr_name: name of the method used to calculate the attribute
            n: number of samples of t spanning the range [0,1]

        Returns:
            None
        """
        if n is None:
            n = self.T
        t_arr = torch.linspace(0, 1, n + 1).double()
        if attr_name in ['alpha', 'sigma', 'SNR', 'SNR_der', 'Karras_t', 'Karras_g_t']:
            attr_val = getattr(self, attr_name)(t_arr)
        elif attr_name == 'logSNR':
            attr_val = self.SNR(t_arr, log=True)
        elif attr_name == 'logSNR_der':
            t_arr = t_arr[1:]  # Avoid t=0 since the derivative can be divergent there
            attr_val = self.attr_der(t_arr, attr='SNR', log=True)
        elif attr_name == 'Karras_dt':
            attr_val = -torch.diff(self.Karras_t(t_arr))
            t_arr = t_arr[:-1]
        elif attr_name == 'Karras_dt/Karras_t':
            Karras_t = self.Karras_t(t_arr)
            attr_val = -torch.diff(Karras_t) / Karras_t[:-1]
            t_arr = t_arr[:-1]
        elif attr_name in ['noise_std']:
            attr_val = getattr(self, attr_name)(t_arr)
            t_arr = t_arr[:-1]
        else:
            raise ValueError(f"attribute {attr_name} is undefined.")

        # Define x and y values
        x_val, y_val = t_arr, attr_val
        xlabel = 't'

        plot_h = plt.plot(x_val, y_val, label=f'{self}')[0]
        if attr_name in ['Karras_t', 'Karras_g_t', 'Karras_dt', 'Karras_dt/sigma', 'SNR', 'noise_std']:
            plt.yscale('log')

        plt.xlabel(xlabel)
        plt.ylabel(attr_name)
        plt.legend()

        return plot_h

    def __repr__(self):
        params_names, params_values = self.extra_params_name + ['T'], []
        for name in params_names:
            param_val = getattr(self, name)
            if isinstance(param_val, torch.Tensor):
                param_val = param_val.item()
            params_values.append(param_val)

        if self.name:
            params_values = [self.name] + params_values
            params_names = ['name'] + params_names
        params_str = ','.join([f"{name}={val}" for name, val in zip(params_names, params_values)])
        # repr = f'{self.name}({params_str},alpha_min={alpha_min},sigma_min={sigma_min})'
        repr = f'{self.__class__.__name__}({params_str})'
        return repr


class PolySch(NoiseSchedule):
    """
    Polynomial schedule where alpha(t) = A t^p + B for a given power p
    A,B are set to ensure alpha(0) = sqrt(1-sigma_min^2) and alpha(1) = alpha_min
    """

    def __init__(self, *args, p=2, **kwargs):
        super().__init__(*args, power=2, **kwargs)
        self.p = nn.parameter.Buffer(torch.tensor(p), persistent=True)
        self.B = torch.sqrt(1 - self.sigma_min.square())
        self.A = self.alpha_min - self.B

    def SNR(self, t: torch.Tensor, log=False):
        alpha_VP = self.A * t.pow(self.p) + self.B
        return self.SNR_alpha(alpha_VP, log=log)


class Poly2Sch(NoiseSchedule):
    """
    Polynomial schedule used in Karras 2022. In Karras, sigma is monotonically DECREASING.
    Here, sigma is decreasing to maintain convention with the other schedules.
    """

    def __init__(self, *args, power=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = nn.parameter.Buffer(torch.tensor(power), persistent=True)
        sigma_max = torch.sqrt(1 - self.alpha_min.square())
        f0 = self.sigma_min ** (1 / self.power)
        f1 = sigma_max ** (1 / self.power)
        self.A = nn.parameter.Buffer(torch.tensor(f1 - f0), persistent=True)
        self.B = nn.parameter.Buffer(torch.tensor(f0), persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        sigma2 = torch.pow(self.A * t + self.B, 2 * self.power)
        alpha_VP = torch.sqrt(1 - sigma2)
        return self.SNR_alpha(alpha_VP, log=log)


class CosineDDPMSch(NoiseSchedule):
    """
    Cosine schedule used in https://arxiv.org/pdf/2102.09672.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def SNR(self, t: torch.Tensor, log=False):
        alpha_VP = torch.cos((t + self.precision) / (1 + self.precision) * torch.pi / 2).abs()
        return self.SNR_alpha(alpha_VP, log=log)


class Cosine2Sch(NoiseSchedule):
    """
    Cosine schedule where alpha(t) = s*cos(f(t))^p where f(t) = At+B with and s=scale and p=power.
    A,B are set to ensure alpha(0) = sqrt(1-sigma_min^2) and alpha(1) = alpha_min
    """

    def __init__(self, *args, power=1.0, scale=1.0, **kwargs):
        super().__init__(power=1.0, scale=1.0, *args, **kwargs)
        alpha_max = torch.sqrt(1 - self.sigma_min.double().square())
        f0 = torch.arccos((alpha_max / scale) ** (1 / power))
        f1 = torch.arccos((self.alpha_min / scale) ** (1 / power))
        if np.isnan(f0) or np.isnan(f1):
            raise ValueError(f"Nans found in defining noise schedule parameters. f0={f0}, f1={f1}")
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)
        self.power = nn.parameter.Buffer(torch.tensor(power), persistent=True)
        self.scale = nn.parameter.Buffer(torch.tensor(scale), persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        f = self.A * t + self.B
        if self.scale == 1.0 and self.power == 1.0:
            Lambda = -2 * self.power * torch.log(torch.tan(f))
        else:
            Lambda = (2 * (self.power * torch.log(torch.cos(f)) + torch.log(self.scale)) -
                      torch.log1p(-self.scale.square() * torch.cos(f) ** (2 * self.power)))

        if log:
            return Lambda
        else:
            return Lambda.exp()


class SigmoidSch(NoiseSchedule):
    """
    Sigmoid schedule where alpha(t) = sigmoid(f(t)),  f(t) = At + B
    A,B are set to ensure alpha(0) = sqrt(1-sigma_min^2) and alpha(1) = alpha_min
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sigmoid_inv = lambda x: torch.log(x) - torch.log1p(-x)
        f0 = sigmoid_inv(torch.sqrt(1 - self.sigma_min.square()))
        f1 = sigmoid_inv(self.alpha_min)
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        # SNR = alpha^2/sigma^2 = sigmoid(f)^2/(1-sigmoid(f)^2) where f = f(t) = At+B
        # sigmoid(f)^2 = 1/(1+exp(-f))^2,    1-sigmoid(f)^2 = 1 - 1/(1+exp(-f))^2 = (2exp(-f)+exp(-2f))/(1+exp(-f))^2
        # sigmoid(f)^2/(1-sigmoid(f)^2) = 1/(2exp(-f)+exp(-2f)) = exp(f)/(2+exp(-f))
        # log(SNR) = f - log(2+exp(-f)) = f - log(2) + log(1/(1+exp(-f)/2)) = f - log(2) + log(1/(1+exp(-f-log(2))))
        f = self.A * t + self.B

        # Alternative methods
        # exp_f = f.exp()
        # SNR_log_alt1 = 2 * (torch.log1p(exp_f) - torch.log1p((-f).exp())) - torch.log1p(2 * exp_f)
        # SNR_log_alt2 = f - torch.log1p(1+(-f).exp())

        log2 = torch.log(2 * t.new_ones(1))
        SNR_log = f - log2 + torch.nn.functional.logsigmoid(f + log2)
        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class HalfSigmoidSch(NoiseSchedule):
    """
    Noise schedule that spans half of a sigmoid where alpha(t) = 2*sigmoid(f(t)) - 1 and f(t) = At + B
    A,B are set to ensure alpha(0) = sqrt(1-sigma_min^2) and alpha(1) = alpha_min
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        halfsigmoid_inv = lambda x: torch.log1p(x) - torch.log1p(-x)
        alpha_max = torch.sqrt(1 - self.sigma_min.square())
        f0 = halfsigmoid_inv(alpha_max)
        f1 = halfsigmoid_inv(self.alpha_min)
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        f = self.A * t + self.B
        exp_f = f.exp()
        SNR_log = 2 * (torch.log1p(exp_f) - torch.log1p((-f).exp())) - torch.log1p(2 * exp_f)
        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class TanhSch(NoiseSchedule):
    """
    Hyperbolic Tangent schedule where alpha(t) = tanh(f(t)), f(t) = At + B
    A,B are set to ensure alpha(0) = sqrt(1-sigma_min^2) and alpha(1) = alpha_min
    """

    def __init__(self, power=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f0 = torch.atanh((1 - self.sigma_min.square()).pow(1 / power / 2))
        f1 = torch.atanh(self.alpha_min.pow(1 / power))
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)
        self.power = nn.parameter.Buffer(torch.as_tensor(power), persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        f = self.A * t + self.B
        if self.power == 1.0:
            # SNR = tanh(f)^2/(1-tanh(f)^2) = tanh(f)^2/sech(f)^2 = sinh(f)^2, sinh(f) = exp(f) (1-exp(-2f))/2
            log2 = torch.log(2 * t.new_ones(1))
            SNR_log = f - log2 + log1mexp(2 * f)
        else:
            raise NotImplementedError('logSNR is not implemented for power != 1.0')

        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class LogSNRLinearSch(NoiseSchedule):
    """
    Schedule such log(SNR(t)) is linear in t. alpha(t) = sqrt(sigmoid(f(t))) where f(t) = A*t + B
    A,B are set to ensure sigma(0) = sigma_min and alpha(1) = alpha_min
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f1 = 2 * torch.log(self.alpha_min) - torch.log1p(-self.alpha_min.square())  # Ensures alpha(1) = alpha_min
        f0 = -(2 * torch.log(self.sigma_min) - torch.log1p(-self.sigma_min.square()))  # Ensures sigma(0) = sigma_min
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        SNR_log = self.A * t + self.B
        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class ExpWeightFixedSch(NoiseSchedule):
    """
    This schedule is designed such that -1/A dlambda(t)/dt = 1/sigma_d^2 + exp(lambda(t)), lambda(t) = log(SNR(t))
    The solution of this ODE is: lambda(t) = -log(exp(f(t)) - sigma_d^2), f(t) = A t/sigma_d^2 + B
    A,B are set to ensure f(0) = SNR_max and f(1)=SNR_min
    This property implies that the loss weights is not needed in Karras' EDM loss since -dlambda(t)/dt = 1/(a*c_out^2)
    One can therefore renormalize the losses with a constant (1/a) for convenience.

    Reference: Kingma et al. 2023,"Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation"
    """

    def __init__(self, sigma_d: torch.Tensor | float, *args, **kwargs):
        super().__init__(*args, sigma_d=sigma_d, **kwargs)
        self.sigma_d = nn.parameter.Buffer(torch.as_tensor(sigma_d), persistent=True)
        self.sigma_d2 = nn.parameter.Buffer(torch.as_tensor(sigma_d.square()), persistent=True)

        # Useful equations:
        # A t/sigma_d^2 + B = log(exp(-lambda) + sigma_d^2) where lambda = log(SNR)
        # To determine A,B, we require f(0) = SNR_max and f(1)=SNR_min
        SNR_min, SNR_max = convert_SNR_range(alpha_min=self.alpha_min, sigma_min=self.sigma_min)

        # Determine B with: B = 2*log(sigma_d) + log(1 + exp(-lambda_max)/ sigma_d^2)
        B = torch.log(self.sigma_d2) + torch.log1p(1 / self.sigma_d2 / SNR_max)

        # Determine A with: A/sigma_d^2 = -lambda_min + log(1 + sigma_d^2 * exp(lambda_min)) - B
        A = (-SNR_min.log() + torch.log1p(self.sigma_d2 * SNR_min) - B) * self.sigma_d2
        self.A = nn.parameter.Buffer(A, persistent=True)
        self.B = nn.parameter.Buffer(B, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        # SNR_log = -log(exp(f(t)) - sigma_d^2) = -f(t) - log(1 - exp(-f + log(sigma_d^2)))
        f = self.A * t / self.sigma_d2 + self.B
        SNR_log = - f - log1mexp(f - torch.log(self.sigma_d2))
        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class LaplaceSch(NoiseSchedule):
    """
    Noise schedule where log(SNR(t)) is Laplace distributed when t is uniformly distributed
    The Laplace distribution is parameterized with mu,b such that pdf(x) = exp(-|x-mu|/b)/2b where x = log(SNR)

    Reference: Hang et al. 2024, "Improved Noise Schedule for Diffusion Training", https://arxiv.org/abs/2407.03297
    """

    def __init__(self, mu=0.0, b=1.0, **kwargs):
        super().__init__(mu=0.0, b=1.0, **kwargs)
        SNR_min, SNR_max = convert_SNR_range(alpha_min=self.alpha_min, sigma_min=self.sigma_min)
        mu = torch.as_tensor(mu)
        b = torch.as_tensor(b)

        B = (1 - (SNR_max / mu.exp()) ** (-1 / b)) / 2
        A = (1 - (SNR_min / mu.exp()) ** (1 / b)) / 2 + B
        self.A = nn.parameter.Buffer(A, persistent=True)
        self.B = nn.parameter.Buffer(B, persistent=True)
        self.mu = nn.parameter.Buffer(mu, persistent=True)
        self.b = nn.parameter.Buffer(b, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        f = self.A * t - self.B
        Lambda = self.b * torch.sign(f) * torch.log1p(-2 * torch.abs(f)) + self.mu
        if log:
            return Lambda
        else:
            return Lambda.exp()


class SigmaLinearSch(NoiseSchedule):
    """
    Noise schedule where sigma(t) = f(t),  f(t) = At + B
    A,B are set to ensure sigma(0) = sigma_min and sigma(1) = sqrt(1-alpha_min^2)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f0 = self.sigma_min
        f1 = torch.sqrt(1 - self.alpha_min.square())
        self.A = nn.parameter.Buffer(f1 - f0, persistent=True)
        self.B = nn.parameter.Buffer(f0, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        sigma = self.A * t + self.B
        SNR_log = torch.log1p(-sigma.square()) - 2 * torch.log(sigma)
        if log:
            return SNR_log
        else:
            return SNR_log.exp()


class DDIMSch(NoiseSchedule):
    """
    Noise schedule used in DDIM model where h(t) = -1/2 beta(t) where h(t) scales the drift term in the SDE: dx = x h(t) dt + g(t) dw_t.
    beta(t) = (beta_max-beta_min)*tau(t) + beta_min and tau(t) linearly interpolates [eps,1]
    Note that since the effective time minimum is eps, h(0) = -1/2*((beta_max-beta_min)*eps + beta_min)

    Reference: https://arxiv.org/abs/2010.02502
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, eps: float = 1e-3, *args, **kwargs):
        assert beta_min > 0, "beta_min must be positive."
        assert beta_max > 0, "B must be negative."
        assert beta_max > beta_min, "beta_max must be greater than beta_min"
        assert 0 < eps < 1, "0 < epsilon < 1 must be satisfied."
        # Define h(t) = A t + B = -1/2*((beta_max-beta_min)*(eps + (1 - eps) * t) + beta_min)
        # A = -1/2*(beta_max-beta_min)*(1 - eps), B=-1/2*((beta_max-beta_min)*eps + beta_min)
        eps = torch.as_tensor(eps)
        beta_d = torch.as_tensor(beta_max - beta_min)
        beta_min = torch.as_tensor(beta_min)

        # Use the formula alpha(t) = exp(int_eps^t{ h(x) dx}) to find alpha_min and sigma_min
        tau_fun = lambda t: eps + (1 - eps) * t
        alpha_fun = lambda t: torch.exp(-1 / 2 * (beta_d / (1 - eps) / 2 * tau_fun(t) ** 2 + beta_min * tau_fun(t)))
        alpha_min = alpha_fun(torch.tensor(1))
        alpha_max = alpha_fun(torch.tensor(0))
        sigma_min = torch.sqrt(1 - alpha_max.square())

        super().__init__(*args, alpha_min=alpha_min, sigma_min=sigma_min, **kwargs)
        self.beta_d = nn.parameter.Buffer(beta_d, persistent=True)
        self.beta_min = nn.parameter.Buffer(beta_min, persistent=True)
        self.eps = nn.parameter.Buffer(eps, persistent=True)

    def SNR(self, t: torch.Tensor, log=False):
        tau = self.eps + (1 - self.eps) * t
        alpha = torch.exp(-1 / 2 * (self.beta_d / (1 - self.eps) / 2 * tau.square() + self.beta_min * tau))
        return self.SNR_alpha(alpha, log=log)


if __name__ == '__main__':
    # # Test equivalence of redefining sampling of time variable in noise schedule
    # T = 1000
    # # sch = PolySch(alpha_min=1e-2, sigma_min=1e-3, power=5)
    # # sch = LogSNRLinearSch(name='coarse', alpha_min=2e-2, sigma_min=0.1)
    # sch = LogSNRLinearSch(name='fine', SNR_min=1 / 5 ** 2, SNR_max=(1 / 2e-3) ** 2, T=T)
    # tau = sch.t_arr
    #
    # # 1) sigma(t), t_i = f(tau_i)
    # t = sch.Karras_t(tau)
    # sigma, sigma_der = t, 1
    # dt = -torch.diff(t)
    # g_t = torch.sqrt(2 * sigma * sigma_der)
    # noise_std = g_t[:-1] * dt.sqrt()
    # drift_scale = g_t[:-1].square() * dt
    #
    # # 2) sigma(t) = f(t), t_i = tau_i
    # t = tau
    # sigma, sigma_der = sch.Karras_t(tau), -sch.attr_der(tau, attr='Karras_t')
    # dt = t.diff()
    # g_t = torch.sqrt(2 * sigma * sigma_der)
    # noise_std2 = g_t[:-1] * dt.sqrt()
    # drift_scale2 = g_t[:-1].square() * dt
    #
    # plt.plot((1 - noise_std2 / noise_std), label='1 - algebraic derivative/numeric derivative')
    # plt.title('Noise std differences')
    # plt.legend()
    # plt.figure()
    # plt.title('Drift scale differences')
    # plt.plot((1 - drift_scale2 / drift_scale), label='1 - algebraic derivative/numeric derivative')
    # # plt.plot(, label='algebraic derivative')
    # # plt.yscale('log')
    # plt.legend()
    # pass

    # Plot various noise schedules
    matplotlib.use('TkAgg')
    schedules = []
    schedules.append(DDIMSch())
    # schedules.append(PolySch(alpha_min=1e-2, sigma_min=1e-3, power=5))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=1.0))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=2.0))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=2.5))
    # schedules.append(LaplaceSch(alpha_min=1e-1, sigma_min=1e-4, b=2.0))
    # schedules.append(LaplaceSch(alpha_min=1e-1, sigma_min=1e-4, b=2.0, mu=2 * torch.log(torch.tensor(10))))
    # schedules.append(LaplaceSch(alpha_min=1e-1, sigma_min=1e-3, b=2.0, mu=2 * torch.log(torch.tensor(10))))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-2, b=3.0, mu=0))
    # schedules.append(LaplaceSch(alpha_min=1e-1, sigma_min=1e-4, b=3.0, mu=2 * torch.log(torch.tensor(10))))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=2.0))
    # schedules.append(LaplaceSch(alpha_min=1e-3, sigma_min=1e-2, b=4.0))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=4.0))
    # schedules.append(LaplaceSch(alpha_min=1e-2, sigma_min=1e-3, b=5.0))
    # schedules.append(LaplaceSch(name='Coarse4', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.01) ** 2, b=1.0))
    # schedules.append(LaplaceSch(name='Coarse4', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.01) ** 2, b=2.0))
    # schedules.append(LaplaceSch(name='Coarse4', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.01) ** 2, b=4.0))
    # schedules.append(SigmoidSch(precision=1e-4))
    # schedules.append(SigmoidSch(name='Sigmoid DDIM', alpha_min=1e-4, sigma_min=5e-3))  # Reproduces DDIM schedule
    # schedules.append(SigmoidSch(alpha_min=1e-1, sigma_min=1e-3))
    # schedules.append(SigmoidSch(name='Sigmoid3, sig=1e-4', alpha_min=1e-2, sigma_min=1e-4))
    # schedules.append(SigmaLinearSch(alpha_min=1e-2, sigma_min=1e-3))
    # schedules.append(TanhSch(alpha_min=1e-2, sigma_min=1e-4))
    # schedules.append(Cosine2Sch(alpha_min=1e-2, sigma_min=1e-3, power=1.0))
    # schedules.append(Cosine2Sch(alpha_min=1e-3, sigma_min=1e-3, power=0.5, name='Cosine p=0.5'))
    # schedules.append(Cosine2Sch(SNR_min=np.exp(-8), SNR_max=np.exp(13)))
    # schedules.append(Cosine2Sch(SNR_min=np.exp(-16), SNR_max=np.exp(5)))
    schedules.append(LogSNRLinearSch(name='coarse VP', alpha_min=2e-2, sigma_min=0.1, mode='VP'))
    schedules.append(LogSNRLinearSch(name='coarse VE', alpha_min=2e-2, sigma_min=0.1, mode='VE'))
    # schedules.append(LogSNRLinearSch(name='coarse 2', SNR_min=(1 / 20.0) ** 2, SNR_max=(1 / 0.01) ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse 3', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.1) ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse 4', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.005) ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse 5', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.1) ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse 4', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.005) ** 2, T=100))
    # schedules.append(LogSNRLinearSch(name='coarse 4 (High T)', SNR_min=(1 / 50.0) ** 2, SNR_max=(1 / 0.005) ** 2, T=10000))
    # schedules.append(LogSNRLinearSch(name='fine', alpha_min=2e-1, sigma_min=2e-3, T=1000))
    # schedules.append(LogSNRLinearSch(name='fine', SNR_min=(1 / 10.0) ** 2, SNR_max=(1 / 0.002) ** 2, T=1000))
    # schedules.append(LogSNRLinearSch(name='fine sampling 2 (T=1000)', SNR_max=1e6, SNR_min=1 / 5 ** 2, T=1000))
    # schedules.append(LogSNRLinearSch(name='fine sampling (T=100)', alpha_min=2e-1, sigma_min=2e-3, T=100))
    # schedules.append(LogSNRLinearSch(name='fine sampling (T=1000,sigma=2e-2)', alpha_min=2e-1, sigma_min=2e-2, T=1000))
    # schedules.append(LogSNRLinearSch(name='fine', SNR_max=1e6, SNR_min=1 / 5 ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse', alpha_min=2e-2, sigma_min=1e-1))
    # schedules.append(LogSNRLinearSch(name='coarse long', SNR_max=1 / 0.01 ** 2, SNR_min=1 / 100 ** 2))
    # schedules.append(LogSNRLinearSch(name='coarse short', SNR_max=1 / 0.5 ** 2, SNR_min=1 / 50 ** 2, T=100))
    # schedules.append(LogSNRLinearSch(name='coarse sampling', SNR_max=1 / 0.1 ** 2, SNR_min=1 / 50 ** 2, T=1000))
    # schedules.append(LogSNRLinearSch(name='coarse 3', SNR_max=1 / 0.1 ** 2, SNR_min=1 / 50 ** 2, T=10000))
    # schedules.append(LogSNRLinearSch(name='logSNR-linear, high sigma', alpha_min=1e-2, sigma_min=1e-3))
    # schedules.append(LogSNRLinearSch(name='logSNR-linear, low sigma', alpha_min=1e-2, sigma_min=5e-4))
    # schedules.append(LogSNRLinearSch(name='logSNR-linear, small sigma', alpha_min=1e-2, sigma_min=1e-4))
    # schedules.append(LogSNRLinearSch(name='logSNR-linear Chroma', SNR_min=np.exp(-7), SNR_max=np.exp(13.5)))

    # Plot various properties of the schedules
    hist_attrs = ['logSNR']
    for attr_name in hist_attrs:
        plt.figure(figsize=(12, 6))
        for sch in schedules:
            sch.plot_attr_dist(attr_name)

    plotted_attrs = ['alpha', 'sigma', 'SNR', 'logSNR_der', 'Karras_t', 'Karras_dt', 'noise_std']
    for attr_name in plotted_attrs:
        plt.figure()
        for sch in schedules:
            sch.plot_attr_timeseries(attr_name)
    plt.show(block=False)
    pass
