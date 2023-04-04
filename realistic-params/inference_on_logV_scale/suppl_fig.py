
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

def likelihood_slice(fig, ax, chg_param_idx, param_start, param_stop, label):


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

        # Make likelihood
        m = Model(y0, 'RK45', t_treat=t_treat, tolerance=tol, p_fn=p_fns[j],
                  limit_of_quant=True)
        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)
        m.set_tolerance(tol)

        # Plot likelihood slices and synthetic data
        param_range = np.linspace(param_start, param_stop, 100)
        lls = []
        params = true_params.copy()
        for mp in param_range:
            params[chg_param_idx] = mp
            lls.append(likelihood(params))

        # Plot log-likelihood slices
        ax.plot(param_range, lls, label="Tol={}, {}".format(tol, labels[j]),
                color=colors[j], ls=lines[j], lw=widths[j])

        # Reset true_params
        true_params[chg_param_idx] = true_params[chg_param_idx]

    ax.set_xlabel(label)
    ax.set_ylabel('Log-likelihood')
    # ax.legend()



def make_figure():

    fig = plt.figure()

    # param_start, param_stop = 1.71 * 10**(-6), 1.91 * 10**(-6)
    # ax = fig.add_subplot(2, 2, 1)
    # chg_param_idx = 0
    # likelihood_slice(fig, ax, chg_param_idx, param_start, param_stop)

    p0 = 0.661
    beta = (1.81 * 10**(-6))
    delta = 7.07
    c = 14.8
    epsilon = 0.9738
    beta, delta, p0, c, epsilon

    param_start = [delta - 0.5, p0 - 0.1, c -0.5, epsilon - 0.1]
    param_stop = [delta + 0.5, p0 + 0.1, c + 0.5, epsilon + 0.1]
    labels = [r'$\delta$', r'$p_0$', r'$c$', r'$\epsilon$']

    for i in range(len(param_start)):
        ax = fig.add_subplot(2, 2, i+1)
        chg_param_idx = i + 1
        likelihood_slice(fig, ax, chg_param_idx, param_start[i], param_stop[i], labels[i])

    # Collect lines
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    lines_unique = [lines[0], lines[1], lines[2]]

    # Add legend
    fig.legend(lines_unique, labels, loc="upper right")
    # fig.legend(lines_unique, labels)

    plt.show()


if __name__ == '__main__':
    make_figure()
