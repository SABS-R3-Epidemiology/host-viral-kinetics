
import math
import matplotlib.pyplot as plt
import numpy as np
import pints
import scipy.integrate


class Model(pints.ForwardModel):
    def __init__(self, y0, solver, step_size=None, tolerance=None, t_treat=1):
        self.y0 = y0
        self.solver = solver
        self.step_size = step_size
        self.tolerance = tolerance
        self.t_treat = t_treat

    def set_step_size(self, step_size):
        """Update the solver step size.

        Parameters
        ----------
        step_size : float
            New step size
        """
        self.step_size = step_size

    def set_tolerance(self, tolerance):
        """Update the solver rtol.

        Parameters
        ----------
        tolerance : float
            New tolerance
        """
        self.tolerance = tolerance

    def n_parameters(self):
        return 5

    def simulate(self, parameters, times):
        beta, delta, p0, c, epsilon = parameters

        def fun(t, y):
            T, I, V = y
            if t < self.t_treat:
                p = p0
            else:
                p = p0 * (1.0 - epsilon)
            d = [-beta * T * V, beta * T * V - delta * I, p * I - c * V]
            return d

        t_range= (0, max(times))

        res = scipy.integrate.solve_ivp(
            fun,
            t_range,
            self.y0,
            t_eval=times,
            method=self.solver,
            rtol=self.tolerance,
            step_size=self.step_size
        )
        y = res.y
        if y.ndim >= 2:
            y = res.y[2]

        return np.log10(y)


def make_figure():

    chg_param_idx = 0

    tolerances = [1e-5, 1e-10]

    fig = plt.figure()

    ll_axes = []

    for j, tol in enumerate(tolerances):

        for i, noise in enumerate([0.0001, 0.001, .01, .1]):
            # Generate data
            y0 = np.asarray([100000, 1.0, 250])
            m = Model(y0, 'RK45', t_treat=0.52)
            p0 = 0.661
            beta = (1.81 * 10**(-6))
            delta = 7.07
            c = 14.8
            epsilon = 0.9738
            true_params = [1.0, 1.0, 0.01, 1.0, 0.5, noise]

            m.set_tolerance(1e-13)
            times = np.linspace(0, 1, 8)
            y = m.simulate(true_params[:-1], times)
            np.random.seed(123)
            y += np.random.normal(0, true_params[-1], len(times))

            dense_times = np.linspace(0, 1, 5000)
            y_true = m.simulate(true_params[:-1], dense_times)

            # Make likelihood
            problem = pints.SingleOutputProblem(m, times, y)
            likelihood = pints.GaussianLogLikelihood(problem)
            m.set_tolerance(tol)

            param_range = np.linspace(.75, 1.5, 100)
            lls = []
            for mp in param_range:
                true_params[chg_param_idx] = mp
                lls.append(likelihood(true_params))

            if j == 0:
                ax = fig.add_subplot(2, 4, i+1)
                ax.scatter(times, y, color='k', s=7.0, label='Data')
                ax.set_xlabel('Time')
                ax.set_ylabel(r'$\log(V(t))$')
                ax.set_title(r'$\sigma={}$'.format(noise))

                m.set_tolerance(1e-10)
                true_params[chg_param_idx] = 1.0
                ax.plot(dense_times, y_true, lw=1.25, color='k', label='Solution')
                ax.legend()

                ax = fig.add_subplot(2, 4, i+5)
                ll_axes.append(ax)
            else:
                ax =ll_axes[i]
            ax.plot(param_range, lls, lw=1.25, label='Tol={}'.format(tol),
                    ls='-' if j == 0 else '--', color='royalblue' if j == 0 else 'k')
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel('Log-likelihood')
            ax.legend()

        fig.set_tight_layout(True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lines = [':', '-']
    colors = ['k', 'royalblue']
    widths = [2, 1.25]
    for j, tol in enumerate(tolerances[::-1]):
        m.set_tolerance(tol)
        y = m.simulate(true_params[:-1], dense_times)
        ax.plot(dense_times, y, label="Tol={}".format(tol), color=colors[j], ls=lines[j], lw=widths[j])
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\log(V(t))$')
    ax.set_title("Forward solutions at true parameter values")
    plt.show()


if __name__ == '__main__':
    make_figure()
