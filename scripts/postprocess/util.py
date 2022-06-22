import matplotlib.pyplot as plt


def plot_settings(family='serif', fontsize=11,
                  markersize=5, linewidth=None, libertine=False):
    """
    Matplotlib settings.
    """
    plt.rc('font', family=family)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('legend', title_fontsize=fontsize)
    plt.rc('lines', markersize=markersize)
    if linewidth is not None:
        plt.rc('lines', linewidth=linewidth)
    if libertine:
        assert family == 'serif'
        plt.rc('text.latex', preamble=r"""\usepackage{libertine}""")
        plt.rc('text.latex', preamble=r"""
                                      \usepackage{libertine}
                                      \usepackage[libertine]{newtxmath}
                                       """)


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height
