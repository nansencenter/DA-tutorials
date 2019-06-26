from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

# Hack to keep line-spacing constant with/out TeX
LE = '\phantom{$\{x_n^f\}_{n=1}^N$}'

txts = [chr(i+97) for i in range(9)]
txts[0] = 'We consider a single cycle of the EnKF,'+\
          'starting with the analysis state at time $(t-1)$.'+LE+'\n'+\
          'The contours are "equipotential" curves of $\|x-\mu_{t-1}\|_{P_{t-1}}$.'+LE
txts[1] = 'The ensemble $\{x_n^a\}_{n=1}^N$ is (assumed) sampled from this distribution.'+LE+'\n'+LE
txts[2] = 'The ensemble is forecasted from time $(t-1)$ to $t$ '+\
          'using the dynamical model $f$.'+LE+'\n'+\
          'We now denote it using the superscript $f$.'+LE
txts[3] = 'Now we consider the analysis at time t. The ensemble is used'+LE+'\n'+\
          'to estimate $\mu^f_t$ and $P^f_t$, i.e. the new contour curves.'+LE
txts[4] = 'The likelihood is taken into account...'+LE+'\n'+LE
txts[5] = "...which implicitly yields this posterior (Bayes' rule)." +LE+'\n'+LE
txts[6] = 'Explicitly, however,'+LE+'\n'+\
          'we compute the Kalman gain, based on the ensemble estimates.'+LE
txts[7] = 'The Kalman gain is then used to shift the ensemble such that it represents' +LE+'\n'+\
          'the (implicit) posterior. The cycle can then begin again, now from $t$ to $t+1$.'+LE

def crop(img):
    top = int(    0.05*img.shape[0])
    btm = int((1-0.08)*img.shape[0])
    lft = int(    0.01*img.shape[1])
    rgt = int((1-0.01)*img.shape[1])
    return img[top:btm,lft:rgt]

def illust_EnKF(i):
    with sns.axes_style("white"):
        plt.figure(1,figsize=(10,12))
        axI = plt.subplot(111)
        axI.set_axis_off()
        axI.set_title(txts[i],loc='left',usetex=True,size=15)
        axI.imshow(crop(imread('./tutorials/resources/illust_EnKF/illust_EnKF_prez_'+str(i+8)+'.png')))
        # Extract text:
        #plt.savefig("images/txts_"+str(i+8)+'.png')
        #bash: for f in `ls txts_*.png`; do convert -crop 800x110+120+260 $f $f; done

# EnKF_animation = interactive(illust_EnKF,i=IntSlider(min=0, max=7,continuous_update=False))

