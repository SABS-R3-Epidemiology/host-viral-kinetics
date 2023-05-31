import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import foldnorm

# making fonts consistent
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}\usepackage{amsfonts}'


def expectation(y):
    """Calculates the expectation of |L-L'|.

    Parameters
    ----------
    t : float
        Sum of (a_i^2/sigma^2)

    Returns
    -------
    float
        Expectation of |L-L'|
    """

    mu = y/2
    sigma = np.sqrt(y)

    return foldnorm.mean(c=mu/sigma, scale=sigma, loc=0)


def ninety_nine_percent(y):
    """Calculates the inverse cumulative distribution function of |L-L'|
    at 0.99.

    Parameters
    ----------
    t : float
        Sum of (a_i^2/sigma^2)

    Returns
    -------
    float
        Inverse cumulative distribution function of |L-L'| at 0.99
    """

    mu = y/2
    sigma = np.sqrt(y)

    return foldnorm.ppf(q=0.99, c=mu/sigma, scale=sigma, loc=0)


def make_figure():
    """Makes the bound figure in the supplementary materials section."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    y = [pow(10, -i) for i in np.linspace(-5, 5, 10000)]
    expect = np.vectorize(expectation)
    percent = np.vectorize(ninety_nine_percent)

    ax.plot(y, expect(y), color="black", ls="-",
            label=r"$y = \mathbb{E} \left[ |D| \right]$")
    ax.fill_between(y, 0, percent(y), alpha=0.2, color="black",
                    label=r"$ P \left[ |D| \leq y \right]$"
                    r"$ \leq 0.99$")
    ax.set_xscale('log')

    # Use a linear scale for y below 10**(-3)
    # (so that the whole shaded region can be shown)
    ax.set_yscale('symlog', linthresh=10**(-3))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\frac{\sum_{i=1}^N a_i^2}{(\sigma^\text{True})^2}$')
    ax.set_ylabel(r'$y$')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    make_figure()
