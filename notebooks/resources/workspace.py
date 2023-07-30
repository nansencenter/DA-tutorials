"""Load tutorials workspace."""

import numpy as np
import matplotlib as mpl
try:
    import google.colab  # type: ignore
    # Colab only supports mpl inline backend => no point loading other.

    # Make figures and fonts larger.
    # Must NOT be in 1st cell of the notebook, because Colab does
    # %matplotlib inline at startup (I think), resetting rcParams.
    mpl.rcParams.update({'font.size': 15})
    mpl.rcParams.update({'figure.figsize': [10,6]})

except ImportError:
    # Should PRECEDE plt.ion()
    mpl.use('nbAgg') # = %matplotlib notebook

    # Note: `%matplotlib inline` is used in tutorials which include `@interact`.
    # because `nbAgg` steals focus from sliders,
    # and re-generates entire figure (not just canvas).

# Load answers
from .answers import answers, show_answer, show_example

# Load widgets
from ipywidgets import interact, Image, interactive, VBox, IntSlider, SelectMultiple

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

def get_jointplotter(grid1d, dx2=1):
    fig, (ax, yax, xax) = axes_with_marginals()
    def plotter(Z, colors=None):
        Z = Z / Z.sum() / dx2
        lvls = np.logspace(-3, 3, 21)
        h = ax.contour(grid1d, grid1d, Z, colors=colors, levels=lvls)
        xax.plot(grid1d, Z.sum(0))
        yax.plot(Z.sum(1), grid1d)
        return h.legend_elements()[0][0]
    return ax, plotter

# TODO:
# @ws.interactive_fig(
#     right=True,
#     right=['corr', 'y2', 'R2']
#     vert=['y2'],
#     corr=(-1, 1, .1),
#     y1=bounds,
#     y2=bounds,
#     R1=(0.01, 20, 0.2),
#     R2=(0.01, 20, 0.2))


####################################
# DA video
####################################
import io
import base64
from IPython.display import HTML
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
  video = io.open('./resources/darc_envisat_analyses.mp4', 'r+b').read()
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


####################################
# EnKF animation
####################################
# Init image
wI = Image(
    value=open("./resources/illust_EnKF/illust_EnKF_0.png", "rb").read(),
    format='png',
    width=800,
    height=600,
)

# Update
def set_image(i=0):
    img = "./resources/illust_EnKF/illust_EnKF_"+str(i)+".png"
    wI.value=open(img, "rb").read()

# Slider
wS = interactive(set_image,i=(0,7,1))

# Stack elements
EnKF_animation = VBox([wS,wI])


####################################
# Misc
####################################
def weave_fa(xf,xa=None):
    "Make piece-wise graph for plotting f/a lines together"
    if xa is None:
        xa = xf
    else:
        assert len(xf)==len(xa)
    # Assemble piece-wise lines for plotting purposes
    pw_f  = np.array([[xa[k  ], xf[k+1], np.nan] for k in range(len(xf)-1)]).ravel()
    pw_a  = np.array([[xf[k+1], xa[k+1], np.nan] for k in range(len(xf)-1)]).ravel()
    return pw_f, pw_a
