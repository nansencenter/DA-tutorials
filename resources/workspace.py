"""Load tutorials workspace."""

# Load nbAgg backend before dapper does plt.ion()
import matplotlib as mpl
mpl.use('nbAgg') # inline (like ipkernel/pylab/backend_inline.py), and interactive.

# Load DAPPER
from dapper import *

# Load answers
from .answers import answers, show_answer, show_example, macros

# Load widgets
from ipywidgets import *

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
wI = Image(
    value=open("./resources/illust_EnKF/illust_EnKF_prez_8.png", "rb").read(),
    format='png',
    width=600,
    height=400,
)
wT = Image(
    value=open("./resources/illust_EnKF/txts_8.png", "rb").read(),
    format='png',
    width=600,
    height=50,
)
def show_image(i=0):
    img = "./resources/illust_EnKF/illust_EnKF_prez_"+str(i+8)+".png"
    txt = "./resources/illust_EnKF/txts_"+str(i+8)+".png"
    wI.value=open(img, "rb").read()
    wT.value=open(txt, "rb").read()
    
wS = interactive(show_image,i=(0,7,1))
EnKF_animation = VBox([wS,wT,wI])



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
    pw_f  = array([[xa[k  ], xf[k+1], nan] for k in range(len(xf)-1)]).ravel()
    pw_a  = array([[xf[k+1], xa[k+1], nan] for k in range(len(xf)-1)]).ravel()
    return pw_f, pw_a




