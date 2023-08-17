"""Load tutorials workspace."""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import mpl_tools


import matplotlib.pyplot as plt
plt.style.use(['seaborn'])


# Should PRECEDE plt.ion()
try:
    # Note: Colab only supports `%matplotlib inline` ⇒ no point loading other.
    # NOTE: Colab: must use plt.show() to avoid duplicate figures.
    import google.colab  # type: ignore
    # Colab only supports mpl inline backend

    # Make figures and fonts larger.
    mpl.rcParams.update({'font.size': 15})
    mpl.rcParams.update({'figure.figsize': [10,6]})
except ImportError:
    if mpl_tools.is_notebook_or_qt:
        # NB: `nbAgg` steals focus from interactive sliders,
        # and re-generates entire figure (not just canvas).
        # mpl.use('nbAgg') # = %matplotlib notebook
        pass # all notebooks use `%matplotlib inline` anyway
    else:
        # Regular python (or ipython) session
        pass

# Must NOT be in 1st cell of the notebook,
# because Colab does %matplotlib inline at startup (I think), resetting rcParams.
mpl.rcParams.update({'lines.linewidth': 2.5})

# Load answers
from .answers import show_answer

# Load widgets
from ipywidgets import Image, interactive, HBox, VBox, IntSlider, SelectMultiple
from IPython.display import display


def interact(top=None, right=None, bottom=None, left=None, **kwargs):
    """Like `ipywidgets.interact(**kwargs)` but with layout shortcuts.

    Set `bottom` or any other `side` argument to `True` to place all controls there,
    relative to the central output (typically figure).
    Otherwise, use a list (or comma-separated string) to select which controls to place there.
    Use *nested* lists to re-group/order them.
    The underlying mechanism is CSS flex box (typically without "wrap").

    If the last element of a `side` is a dict, then it will be written as attributes
    to the CSS `layout` attribute, ref [1].
    Support for the `style` attribute [2] is not yet implemented.

    Similarly, if the last element of any `kwargs` is a dict, then it will be written as attributes
    (e.g. `description (str)`, 'readout (bool)', `continuous_update (bool)`, `orientation (str)`)
    to the widget, ref [3].

    Only tested with "inline" backend (Colab and locally).
    Also see `~/P/HistoryMatching/tools/plotting.py`

    [1]: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Layout.html
    [2]: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Styling.html
    [3]: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#

    Example:

    >>> v = dict(orientation="vertical", layout=dict(height="80%"))
    ... @ws.interact(a=(1., 6., v),
    ...              b=(1., 7.),
    ...              bottom=True,  # put rest here
    ...              top='b,c',
    ...              right=[['a', dict(height="100%", align_items="center")],['e']])
    ... def f(a=3.0, b=4, c=True, d=5, e=6):
    ...     plt.figure(figsize=(4, 5))
    ...     xx = np.linspace(0, 3, 21)
    ...     if c: plt.plot(xx, e*d/a + xx**b)
    ...     else: plt.plot(xx, b + xx)
    ...     plt.show()
    """

    def get_dict(iterable):
        if iterable and isinstance(iterable[-1], dict):
            return iterable[-1]
        else:
            return {}

    def boxit(ww, horizontal=True):
        """Apply box to lists, recursively (alternating between `HBox` and `VBox`)."""
        if (layout := get_dict(ww)):
            ww = ww[:-1]

        for i, w in enumerate(ww):
            if hasattr(w, '__iter__'):
                ww[i] = boxit(w, not horizontal)

        box = HBox if horizontal else VBox
        return box(ww, layout=layout)

    def pop_widgets(ww, labels):
        """Replace items in nested list `labels` by matching elements from `ww`.

        Essentially `[ww.pop(i) for i, w in enumerate(ww) if w.description == lbl]`
        but if `w` is a list, then recurse.
        """
        # Validate
        if not labels:
            return []
        elif labels == True:
            cp = ww.copy()
            ww.clear()
            return cp
        elif isinstance(labels, str):
            labels = labels.split(',')

        # Main
        ww2 = []
        for lbl in labels:
            if isinstance(lbl, dict):
                # Forward as is
                w = lbl
            elif isinstance(lbl, list):
                # Recurse
                w = pop_widgets(ww, lbl)
            else:
                # Pop
                i = [i for i, w in enumerate(ww) if w.description == lbl]
                try:
                    i = i[0]
                except IndexError:
                    raise IndexError(f'Did you specify {lbl} twice in the layout?')
                w = ww.pop(i)
            ww2.append(w)
        return ww2

    sides = dict(top=top, right=right, bottom=bottom, left=left)

    # Pop attributes (if any) for controls
    attrs = {}
    for key, iterable in kwargs.items():
        if (dct := get_dict(iterable)):
            attrs[key] = dct
            kwargs[key] = type(iterable)(iterable[:-1])  # preserve list or tuple

    def decorator(fun):
        # Auto-parse kwargs, add 'observers'
        linked = interactive(fun, **kwargs)
        *ww, out = linked.children
        # display(HBox([out, VBox(ww)]))

        # Styling of individual control widgets
        for w in ww:
            for attr, val in attrs.get(w.description, {}).items():
                setattr(w, attr, val)
            # Defaults
            w.style.description_width = "max-content"
            if getattr(w, 'orientation', '') == "vertical":
                w.layout.width = "2em"

        on = {side: pop_widgets(ww, labels) for side, labels in sides.items()}
        on['right'] = ww + on['right']  # put any remainder on the right (before any dict)

        # Dashbord composition
        # I considered AppLayout, but was more comfortable with combining boxes
        left = boxit(on['left'], False)
        right = boxit(on['right'], False)
        top = boxit(on['top'], True)
        bottom = boxit(on['bottom'], True)

        dashboard = VBox([top, HBox([left, out, right]), bottom])

        display(dashboard);
        linked.update()  # necessary on Colab

    if interact.disabled:
        # Used with hacky `import_from_nb`
        return (lambda fun: (lambda _: None))
    elif not mpl_tools.is_notebook_or_qt:
        # Return dummy (to plot without interactivity)
        return (lambda fun: fun())
    else:
        return decorator

interact.disabled = False


def cInterval(mu, sigma2, flat=True):
    """Compute +/- 1-sigma (std.dev.) confidence/credible intervals (CI)."""
    s1 = np.sqrt(sigma2)
    a = mu - s1
    b = mu + s1
    if flat:
        return a.flatten(), b.flatten()
    else:
        return a, b


def axes_with_marginals():
    from matplotlib import pyplot as plt
    fig, ((ax, yax), (xax, _)) = plt.subplots(
        2, 2, sharex='col', sharey='row',
        figsize=(6, 6),
        gridspec_kw={'height_ratios':[5,1],
                     'width_ratios' :[5,1],
                     'wspace': .1,
                     'hspace': .1})
    _.set_visible(False)
    ax.set_aspect('equal')
    return fig, (ax, yax, xax)


def get_jointplotter(grid1d):
    fig, (ax, yax, xax) = axes_with_marginals()
    dx = grid1d[1] - grid1d[0]
    def plotter(Z, colors=None, alpha=.3, linewidths=1, **kwargs):
        Z = Z / Z.sum() / dx**2
        lvls = np.logspace(-3, 3, 21)
        # h = ax.contourf(grid1d, grid1d, Z, colors=colors,  levels=lvls, alpha=alpha)
        # _ = ax.contour(grid1d, grid1d, Z, colors='black', levels=lvls, linewidths=.7, alpha=alpha)
        h = ax.contour(grid1d, grid1d, Z, colors=colors,  levels=lvls, linewidths=linewidths, **kwargs)

        margx = dx * Z.sum(0)
        margy = dx * Z.sum(1)
        xax.fill_between(grid1d, margx, color=colors, alpha=alpha)
        yax.fill_betweenx(grid1d, 0, margy, color=colors, alpha=alpha)

        return h.legend_elements()[0][0]
    return ax, plotter


def frame(data, ax, zoom=1):
    """Do `ax.set_{x/y/z}lim()` based on `data`, using given `zoom` (power of 10)."""
    zoom = 10**(zoom - 1)
    for ens, dim in zip(data.T, 'xyz'):
        a = ens.min()
        b = ens.max()
        m = (a + b)/2
        w = b - a
        setter = getattr(ax, f'set_{dim}lim')
        setter([m - w/2/zoom,
                m + w/2/zoom])


def envisat_video():
    caption = """Illustration of DA for the ozone layer in 2002.
    <br><br>
    LEFT: Satellite data (i.e. all that is observed).
    RIGHT: Simulation model with assimilated data.
    <br><br>
    Could you have perceived the <a href='http://dx.doi.org/10.1175/JAS-3337.1'>splitting of the ozone hole.</a> only from the satellite data?
    <br><br>
    Attribution: William A. Lahoz, DARC.
    """

    import io
    import base64
    from IPython.display import HTML

    video = io.open(Path(__file__).parent / 'darc_envisat_analyses.mp4', 'r+b').read()
    encoded = base64.b64encode(video)
    vid = HTML(data='''
    <figure style="width:580px;">
    <video alt="{1}" controls style="width:550px;">
    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
    </video>
    <figcaption style="background-color:#d9e7ff;">{1}</figcaption>
    </figure>
    '''.format(encoded.decode('ascii'),caption))
    return vid


def EnKF_animation():
    # Initialize
    path_ = str(Path(__file__).parent / "illust_EnKF/illust_EnKF_")
    image = Image(
        value=open(path_ + "0.png", "rb").read(),
        format='png',
        width=800,
        height=600,
    )

    def update_image(i=0):
        image.value=open(path_ + str(i) + ".png", "rb").read()

    slider = interactive(update_image, i=(0, 7, 1))
    return VBox([slider, image])


# TODO: use pre-commit setup suggested by jupytext to ensure nb's are synced?
def import_from_nb(name: str, objs: list):
    """Import `objs` from `notebooks/name*.py` (1st match).

    Might not want to do this because it uses `sys.path` manipulation,
    imposes that the notebook contain

    - only light computations (unless controlled by interact.disabled)
    - the version we want (as opposed to it being implemented by students)

    and because a little repetition never hurt nobody.
    """
    NBDIR = Path(__file__).parents[1]
    notebk = next(NBDIR.glob(name + "*.ipynb"))
    script = (NBDIR / "scripts" / notebk.relative_to(NBDIR)).with_suffix('.py')

    interact.disabled = True
    try:
        name = str(script.relative_to(NBDIR).with_suffix("")).replace("/", ".")
        script = getattr(__import__(name), script.stem)  # works despite weird chars
    finally:
        interact.disabled = False
    return [getattr(script, x) for x in objs]
