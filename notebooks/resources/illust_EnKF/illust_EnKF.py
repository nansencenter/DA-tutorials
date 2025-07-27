"""Script to patch png figures
from Matlab script DATUM/illust_EnKF_1.m
together with text titles, as given below.
"""

from pathlib import Path

import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

from matplotlib.image import imread
import matplotlib.pyplot as plt

plt.ion()

txts = []
txts += [
    "We consider a single cycle of the EnKF, "
    "starting with the analysis state\n"
    "at time $(k-1)$. "
    "The contours are level curves of "
    "$\|\mathbf{x}-\mathbf{\\bar{x}}^{a}_{k-1}\|_{\mathbf{\\bar{P}}^{a}_{k-1}}$..."
]
txts += [
    "We consider a single cycle of the EnKF, "
    "starting with the analysis ensemble\n"
    "at time $(k-1)$, $\{\mathbf{x}_n^{a}\}_{n=1..N}$, "
    "with mean and cov. estimates $\mathbf{\\bar{x}}^{a}_{k-1}$ and $\mathbf{\\bar{P}}^{a}_{k-1}$.\n"
    "The contours are level curves of "
    "$\|\mathbf{x}-\mathbf{\\bar{x}}^{a}_{k-1}\|_{\mathbf{\\bar{P}}^{a}_{k-1}}$.",
]
txts += [
    "The ensemble is forecasted from time $(k-1)$ to $k$ "
    "by the dyn. "
    "model $\mathscr{M}$.\n We now denote it using the superscript $f$."
]
txts += [
    "Now we consider the analysis at time $k$. The ensemble \emph{could} be used\n"
    "to compute the estimates $\mathbf{\\bar{x}}^{f}_k$ and $\mathbf{\\bar{P}}^{f}_k$, "
    "hence the new contour curves."
]
txts += ["Whereupon an obs. likelihood..."]
txts += ["...\emph{would} yield a posterior by Bayes' rule."]
txts += [
    "What we \emph{equivalently} do instead,\nis to compute the Kalman gain "
    "using the estimate $\mathbf{\\bar{P}}^{f}_k$."
]
txts += [
    "The Kalman gain is then used to shift the ensemble.\n"
    "We know that it gets shifted to where the (implicit) posterior lies.\n"
    "The cycle can then begin again, from $k$ to $k+1$."
]

# Hack to keep line-spacing constant with/out TeX
placeholder = "\phantom{$\{x_n^f\}_{n=1}^N$}"
placeholder += "."  # phantom w/o anything causes stuff to disappear
for i, t in enumerate(txts):
    t = t.split("\n")
    t = [placeholder] * (2 - len(t)) + t  # ensure 2 lines
    # t = [ln+LE for ln in t]
    txts[i] = "\n".join(t)


def crop(img):
    "Crop Matlab-outputted image"
    top = int(0.15 * img.shape[0])
    btm = int((1 - 0.20) * img.shape[0])
    lft = int(0.10 * img.shape[1])
    rgt = int((1 - 0.09) * img.shape[1])
    return img[top:btm, lft:rgt]


PWD = Path(__file__).parent


def illust_EnKF(i):
    plt.close(1)
    plt.figure(1, figsize=(8, 6))
    axI = plt.subplot(111)
    axI.set_axis_off()
    name = "illust_EnKF_prez_" + str(i + 8) + ".png"
    name = PWD / "from_Matlab" / name
    img = imread(name)
    img = crop(img)
    axI.imshow(img)
    axI.set_title(txts[i], loc="left", usetex=True, size=15)


for i, txt in enumerate(txts):
    illust_EnKF(i)
    plt.pause(0.2)
    name = "illust_EnKF_" + str(i) + ".png"
    print("Saving", PWD / name)
    plt.savefig(PWD / name)
