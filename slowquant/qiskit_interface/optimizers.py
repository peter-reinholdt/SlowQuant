from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import scipy
from scipy.optimize import minimize_scalar


class Result:
    """Result class for optimizers."""

    def __init__(self) -> None:
        """Initialize result class."""
        self.x: np.ndarray
        self.fun: float


class RotoSolve:
    r"""Rotosolve optimizer.

    Implemenation of Rotosolver assuming three eigenvalues for generators.
    This works for fermionic generators of the type:

    .. math::
        \hat{G}_{pq} = \hat{a}^\dagger_p \hat{a}_q - \hat{a}_q^\dagger \hat{a}_p

    and,

    .. math::
        \hat{G}_{pqrs} = \hat{a}^\dagger_p \hat{a}^\dagger_q \hat{a}_r \hat{a}_s - \hat{a}^\dagger_s \hat{a}^\dagger_r \hat{a}_p \hat{a}_q

    Rotosolve works by exactly reconstrucing the energy function in a single parameter:

    .. math::
        E(x) = \frac{\sin\left(\frac{2R+1}{2}x\right)}{2R+1}\sum_{\mu=-R}^{R}E(x_\mu)\frac{(-1)^\mu}{\sin\left(\frac{x - x_\mu}{2}\right)}

    With :math:`R` being the number of unique absolute eigenvalues, i.e. -1 and 1 is the "same" eigenvalue in this context, and :math:`x_\mu=\frac{2\mu}{2R+1}\pi`.
    In this implementation it is assumed that :math:`R=2`.

    After the function :math:`E(x)` have been reconstruced the global minima of the function can be found classically.

    #. 10.22331/q-2021-01-28-391, Algorithm 1
    #. 10.22331/q-2022-03-30-677, Eq. (57)
    """

    def __init__(
        self,
        R: dict[str, int],
        param_names: Sequence[str],
        maxiter: int = 30,
        tol: float = 1e-6,
        line_search: bool = False,
        callback: Callable[[list[float]], None] | None = None,
    ) -> None:
        """Initialize Rotosolver.

        Args:
            R: R parameter used for the function reconstruction.
            param_names: Names of parameters, used to index R.
            maxiter: Maximum number of iterations (sweeps).
            tol: Convergence tolerence.
            line_search: Do a line search after each rotosolve iteration
            callback: Callback function, takes only x (parameters) as an argument.
        """
        self._callback = callback
        self.max_iterations = maxiter
        self.threshold = tol
        self.max_fail = 3
        self._R = R
        self.line_search = line_search
        self._param_names = param_names

    def minimize(
        self, f: Callable[[list[float]], float], x: list[float], jac=None  # pylint: disable=unused-argument
    ) -> Result:
        """Run minimization.

        Args:
            f: Function to be minimzed, can only take one argument.
            x: Changable parameters of f.
            jac: Placeholder for gradient function, is not used.

        Returns:
            Minimization results.
        """
        f_best = float(10**20)
        x_best = x.copy()
        fails = 0
        res = Result()
        x = np.array(x)
        for _ in range(self.max_iterations):
            x_old = np.copy(x)
            for i, par_name in enumerate(self._param_names):
                e_vals = get_energy_evals(f, x, i, self._R[par_name])
                f_reconstructed = partial(reconstructed_f, energy_vals=e_vals, R=self._R[par_name])
                vecf = np.vectorize(f_reconstructed)
                values = vecf(np.linspace(-np.pi, np.pi, int(1e4)))
                theta = np.linspace(-np.pi, np.pi, int(1e4))[np.argmin(values)]
                res = scipy.optimize.minimize(f_reconstructed, x0=[theta], method="BFGS", tol=1e-12)
                x[i] = res.x[0]
                period = 2*np.pi
                if np.allclose(f_reconstructed(x[i]), f_reconstructed(x[i]+np.pi)):
                    period = np.pi
                while np.abs(x[i] + period) < np.abs(x[i]):
                    x[i] += period
                while np.abs(x[i] - period) < np.abs(x[i]):
                    x[i] -= period
            f_new = f(x)
            if self.line_search:
                delta_x = x - x_old
                x_old = np.copy(x)
                f_line = lambda s: f(x_old+s*delta_x)
                res_line = minimize_scalar(f_line, (0, 2))
                f_min = res_line.fun
                s_min = res_line.x
                if f_min < f_new:
                    f_new = f_min
                    x = x_old + s_min * delta_x
            if abs(f_best - f_new) < self.threshold:
                f_best = f_new
                x_best = x.copy()
                break
            if (f_new - f_best) > 0.0:
                fails += 1
            else:
                f_best = f_new
                x_best = x.copy()
            if fails == self.max_fail:
                break
            if self._callback is not None:
                self._callback(x)
        res.x = np.array(x_best)
        res.fun = f_best
        return res


def get_energy_evals(f: Callable[[list[float]], float], x: list[float], idx: int, R: int) -> list[float]:
    """Evaluate the function in all points needed for the reconstrucing in Rotosolve.

    Args:
        f: Function to evaluate.
        x: Parameters of f.
        idx: Index of parameter to be changed.
        R: Parameter to control how many points are needed.

    Returns:
        All needed function evaluations.
    """
    e_vals = []
    x = x.copy()
    for mu in range(-R, R + 1):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        x[idx] = x_mu
        e_vals.append(f(x))
    return e_vals


def reconstructed_f(x: float, energy_vals: list[float], R: int) -> float:
    r"""Reconstructed the function in terms of sin-functions.

    .. math::
        E(x) = \frac{\sin\left(\frac{2R+1}{2}x\right)}{2R+1}\sum_{\mu=-R}^{R}E(x_\mu)\frac{(-1)^\mu}{\sin\left(\frac{x - x_\mu}{2}\right)}

    For better numerical stability the implemented form is instead:

    .. math::
        E(x) = \sum_{\mu=-R}^{R}E(x_\mu)\frac{\mathrm{sinc}\left(\frac{2R+1}{2}(x-x_\mu)\right)}{\mathrm{sinc}\left(\frac{1}{2}(x-x_\mu)\right)}

    #. 10.22331/q-2022-03-30-677, Eq. (57)
    #. https://pennylane.ai/qml/demos/tutorial_general_parshift/, 2024-03-14

    Args:
        x: Function variable.
        energy_vals: Pre-calculated points of original function.
        R: Parameter to control how many points are needed.

    Returns:
        Function value in x.
    """
    e = 0.0
    for i, mu in enumerate(list(range(-R, R + 1))):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        e += (
            energy_vals[i]
            * np.sinc((2 * R + 1) / 2 * (x - x_mu) / np.pi)
            / (np.sinc(1 / 2 * (x - x_mu) / np.pi))
        )
    return e
