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

def sd(y):
    """Calculates the standard deviation of |L-L'|.
    
    Parameters
    ----------
    t : float
        Sum of (a_i^2/sigma^2)

    Returns
    -------
    float
        Standard deviation of |L-L'|
    """

    mu = y/2
    sigma = np.sqrt(y)

    return np.sqrt(foldnorm.var(c=mu/sigma, scale=sigma, loc=0))

def make_figure():
    """Makes the bound figure in the supplementary materials section."""

    # Create plot of mean and mean +/- 3 * standard deviation
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    y = [pow(10, -i) for i in np.linspace(-5, 5, 10000)]
    expect = np.vectorize(expectation)
    standard_dev = np.vectorize(sd)

    ax.plot(y, expect(y), color="black", ls="-",
            label=r"$\mathbb{E} \left[ |\mathcal{L} - \mathcal{L}' | \right]$")
    ax.fill_between(y, expect(y) - 3 * standard_dev(y),
                    expect(y) + 3 * standard_dev(y), alpha =0.2, color="black",
                    label=r"$\mathbb{E} \left[ |\mathcal{L} - \mathcal{L}' | \right]$"
                     r"$ \pm 3 \sqrt{\text{Var} \left[ |\mathcal{L}-\mathcal{L}'| \right]}$")
    ax.set_xscale('log')

    # Use a linear scale for y below 10**(-3) (so that the whole shaded region can be shown)
    ax.set_yscale('symlog', linthresh=10**(-3))
    ax.set_ylim(bottom = 10**(-1000))
    ax.set_xlabel(r'$\frac{\sum_{i=1}^N a_i^2}{2\sigma^2}$')
    ax.legend()


    plt.show()


if __name__ == '__main__':
    make_figure()
