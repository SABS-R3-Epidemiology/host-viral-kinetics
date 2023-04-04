
import matplotlib.pyplot as plt
import numpy as np
import pints
import scipy.integrate

# making fonts consistent
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = \
r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'


def production_rate_step(t, t_treat, p0, epsilon):
    "Step function production rate, p."

    if t < t_treat:
        p = p0
    else:
        p = p0 * (1.0 - epsilon)

    return p


def production_rate_tanh(t, t_treat, T_max, p0, epsilon):
    "Hyperbolic tan production rate."

    arg = (t-(t_treat + 0.5 * T_max))/(0.25 * T_max)
    p = - (p0 * epsilon/2) * np.tanh(arg) + p0 - (p0 * epsilon/2)

    return p


class Model(pints.ForwardModel):
    def __init__(self, y0, solver, step_size=None, tolerance=None,
                 t_treat=2.775, p_fn="Step", limit_of_quant=True):
        self.y0 = y0
        self.solver = solver
        self.step_size = step_size
        self.tolerance = tolerance
        self.t_treat = t_treat
        self.p_fn = p_fn
        self.limit_of_quant = limit_of_quant

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

            if self.p_fn == "Step":
                p = production_rate_step(t, self.t_treat, p0, epsilon)

            elif self.p_fn == "Tanh":
                t_treat = 2.775
                T_max = 1.08/24
                p = production_rate_tanh(t, t_treat, T_max, p0, epsilon)

            d = [-beta * T * V, beta * T * V - delta * I, p * I - c * V]
            return d

        t_range = (0, max(times))

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

        if self.limit_of_quant is True:
            # limit of quantification
            y[y < 10**(0.7)] = 10**(0.7)

        return np.log10(y)


def make_figure():

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)

    # Make plot of production rates
    p0 = 0.661
    epsilon = 0.9738
    t_treat = 2.775
    T_max = 1.08/24
    t = np.linspace(2.5, 3, 1000)
    p_step = np.vectorize(production_rate_step)
    p_tanh = np.vectorize(production_rate_tanh)
    ax.plot(t, p_step(t, t_treat, p0, epsilon), color="black",
            label="Step")
    ax.plot(t, p_tanh(t, t_treat, T_max, p0, epsilon), color="red", ls='-',
            label="Tanh")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Production rate, $p$')
    ax.legend()

    chg_param_idx = 0

    tolerances = [1e-3, 1e-3, 1e-8]
    p_fns = ["Step", "Tanh", "Step"]

    # Define styles
    lines = ['-', ':', '--']
    colors = ['black', 'red', 'royalblue']
    widths = [1, 1.25, 2]
    labels = ["Step", "Tanh", "Step"]

    # Define parameters
    T_0 = 4 * 10**8
    I_0 = 10**(-6)
    V_0 = 10**(-3.42)
    y0 = np.asarray([T_0, I_0, V_0])
    t_treat = 2.775
    noise = 0.1
    p0 = 0.661
    beta = (1.81 * 10**(-6))
    delta = 7.07
    c = 14.8
    epsilon = 0.9738
    true_params = [beta, delta, p0, c, epsilon, noise]

    sim_axes = []
    ll_axes = []

    for j, tol in enumerate(tolerances):

        # Generate synthetic data (at tolerance 10^(-13))
        m = Model(y0, 'RK45', t_treat=t_treat, p_fn=p_fns[j], limit_of_quant=False)
        m.set_tolerance(1e-13)
        times = np.linspace(0, 7, 8)
        y = m.simulate(true_params[:-1], times)
        np.random.seed(123)
        y += np.random.normal(0, true_params[-1], len(times))

        # apply the limit of quantification (in case the error is negative)
        y[y < 0.7] = 0.7

        # Simulate the forward solution
        m = Model(y0, 'RK45', t_treat=t_treat, tolerance=tol, p_fn=p_fns[j])
        dense_times = np.linspace(0, 7, 5000)
        y_true = m.simulate(true_params[:-1], dense_times)

        if j == 0:
            # Save axes
            ax = fig.add_subplot(1, 3, 2)
            sim_axes.append(ax)
        else:
            # reuse axes
            ax = sim_axes[0]

        # Plot synthetic data
        # ax.scatter(times, y, s=7.0, label='Data', color=colors[j])
        ax.scatter(times, y, s=7.0, color=colors[j])

        # Plot solution
        # ax.plot(dense_times, y_true, label="Tol={}".format(tol),
        #         color=colors[j], ls=lines[j], lw=widths[j])
        ax.plot(dense_times, y_true, label="tol={}, {}".format(tol, labels[j]),
                color=colors[j], ls=lines[j], lw=widths[j])
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\log_{10}(V(t))$')
        # ax.set_title("Forward solutions at true parameter values")

        if j == 0:
            # Save axes
            ax = fig.add_subplot(1, 3, 3)
            ll_axes.append(ax)

        else:
            # reuse axes
            ax = ll_axes[0]

        # Make likelihood
        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)
        m.set_tolerance(tol)

        # Plot likelihood slices and synthetic data
        # param_range = np.linspace(1.80 * 10**(-6), 1.82 * 10**(-6), 100)
        # param_range = np.linspace(1.76 * 10**(-6), 1.86 * 10**(-6), 100)
        param_range = np.linspace(1.71 * 10**(-6), 1.91 * 10**(-6), 100)
        lls = []
        for mp in param_range:
            true_params[chg_param_idx] = mp
            lls.append(likelihood(true_params))

        # Plot log-likelihood slices
        # ax.plot(param_range, lls, label='Tol={}'.format(tol),
        #         color=colors[j], ls=lines[j], lw=widths[j])
        ax.plot(param_range, lls, label="Tol={}, {}".format(tol, labels[j]),
                color=colors[j], ls=lines[j], lw=widths[j])

        # Reset true_params
        true_params[chg_param_idx] = 1.81 * 10**(-6)

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Log-likelihood')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    make_figure()
