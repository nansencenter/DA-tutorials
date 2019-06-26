from markdown import markdown as md2html # better than markdown2 ?
from IPython.display import HTML, display

# Notes:
# - md2html rendering sometimes breaks
#   because it has failed to parse the eqn properly.
#   For ex: _ in math sometimes gets replaced by <em>.
#   Can be fixed by escaping, i.e. writing \_

def formatted_display(TYPE,s,bg_color):
    s = s[1:] # Remove newline
    bg = 'background-color:'+ bg_color + ';' #d8e7ff #e2edff
    if   TYPE == "HTML": s = s
    elif TYPE == "MD"  : s = md2html(s)
    elif TYPE == "TXT" : s = '<code style="'+bg+'">'+s+'</code>'
    s = ''.join([
        '<div ',
        'style="',bg,'padding:0.5em;">',
        str(s),
        '</div>'])
    display(HTML(s))

def show_answer(tag):
    formatted_display(*answers[tag], '#dbf9ec') # #d8e7ff

        
def show_example(tag):
    formatted_display(*examples[tag], '#ffed90')


answers = {}
examples = {}

macros=r'''%
%MACRO DEFINITION
\newcommand{\Reals}{\mathbb{R}}
\newcommand{\Imags}{i\Reals}
\newcommand{\Integers}{\mathbb{Z}}
\newcommand{\Naturals}{\mathbb{N}}
%
\newcommand{\Expect}[0]{\mathop{}\! \mathbb{E}}
\newcommand{\NormDist}{\mathop{}\! \mathcal{N}}
%
\newcommand{\mat}[1]{{\mathbf{{#1}}}} 
%\newcommand{\mat}[1]{{\pmb{\mathsf{#1}}}}
\newcommand{\bvec}[1]{{\mathbf{#1}}}
%
\newcommand{\trsign}{{\mathsf{T}}}
\newcommand{\tr}{^{\trsign}}
%
\newcommand{\I}[0]{\mat{I}}
\newcommand{\K}[0]{\mat{K}}
\newcommand{\bP}[0]{\mat{P}}
\newcommand{\F}[0]{\mat{F}}
\newcommand{\bH}[0]{\mat{H}}
\newcommand{\bF}[0]{\mat{F}}
\newcommand{\R}[0]{\mat{R}}
\newcommand{\Q}[0]{\mat{Q}}
\newcommand{\B}[0]{\mat{B}}
\newcommand{\Ri}[0]{\R^{-1}}
\newcommand{\Bi}[0]{\B^{-1}}
\newcommand{\X}[0]{\mat{X}}
\newcommand{\A}[0]{\mat{A}}
\newcommand{\Y}[0]{\mat{Y}}
\newcommand{\E}[0]{\mat{E}}
\newcommand{\U}[0]{\mat{U}}
\newcommand{\V}[0]{\mat{V}}
%
\newcommand{\x}[0]{\bvec{x}}
\newcommand{\y}[0]{\bvec{y}}
\newcommand{\q}[0]{\bvec{q}}
\newcommand{\br}[0]{\bvec{r}}
\newcommand{\bb}[0]{\bvec{b}}
%
\newcommand{\cx}[0]{\text{const}}
\newcommand{\norm}[1]{\|{#1}\|}
%
\newcommand{\bx}[0]{\bvec{\bar{x}}}
\newcommand{\barP}[0]{\mat{\bar{P}}}
%'''

###########################################
# Tut: DA & EnKF
###########################################
answers['thesaurus 1'] = ["TXT",r"""
Data Assimilation (DA)     Ensemble      Stochastic     Data        
Filtering                  Sample        Random         Measurements
Kalman filter (KF)         Set of draws  Monte-Carlo    Observations
State estimation           
Data fusion                
"""]

answers['thesaurus 2'] = ["TXT",r"""
Statistical inference    Ensemble member     Quantitative belief    Recursive 
Inverse problems         Sample point        Probability            Sequential
Inversion                Realization         Relative frequency     Iterative 
Estimation               Single draw                                Serial    
Approximation            Particle
Regression               
Fitting                  
"""]


###########################################
# Tut: Bayesian inference
###########################################
answers['pdf_G1'] = ['MD',r'''
    pdf_values = 1/sqrt(2*pi*B)*exp(-0.5*(x-b)**2/B)
    # Version using the scipy (sp) library:
    # pdf_values = sp.stats.norm.pdf(x,loc=b,scale=sqrt(B))
''']

examples['BR'] = ['MD',r'''
 - You believe the temperature $(x)$ in the room is $22°C \pm 2°C$;  
more specifically, your prior is: $p(x) = \mathcal{N}(x \mid 22, 4)$.  
 - A thermometer yields the observation $y = 24°C \pm 2°C$;  
more specifically, the likelihood is: $p(y|x) = \mathcal{N}(24 \mid x, 4)$.  
 - Your updated, posterior belief is then $p(x|y) = \mathcal{N}(x \mid 23, 2)$.  
(exactly how these numbers are calculated will be shown below).
''']

answers['BR derivation'] = ['MD',r'''
<a href="https://en.wikipedia.org/wiki/Bayes%27_theorem#Derivation" target="_blank">Wikipedia</a>

''']

answers['inverse'] = ['MD',r'''
Because estimation (i.e. inference) is seen as reasoning backwards from the "outcome" to the "cause".
In physics, causality is a difficult notion.
Still, we use it to define the direction "backward", or "inverse",
which we associate with the symolism $f^{-1}$.
Since $y = f(x)$ is common symbolisism,
it makes sense to use the symobls $x = f^{-1}(y)$ for the estimation problem.
''']

answers['Posterior behaviour'] = ['MD',r'''
 - Likelihood becomes flat.  
 Posterior is dominated by the prior, and becomes (in the limit) superimposed on it.
 - Likelihood becomes a delta function.  
 Posterior is dominated by the likelihood, and becomes superimposed on it.
 - It's located halfway between the prior/likelihood $\forall y$.
 - No.
 - No (in fact, we'll see later that it remains Gaussian,
 and is therefore fully characterized by its mean and its variance).
 - It would seem we only need to compute its location and scale
 (otherwise, the shape remains unchanged).
''']

answers['BR normalization'] = ['MD',r'''
$$\texttt{sum(pp)*dx}
\approx \int \texttt{pp}(x) \, dx
= \int p(x) \, p(y|x) \, dx
= \int p(x,y) \, dx
= p(y) \, .$$


<!--
*Advanced*:
Firstly, note that normalization is quite necessary, being requried by any expectation computation. For example, $\mathbb{E}(x|y) = \int x \, p(x) \, dx \approx$ `x*pp*dx` is only valid if `pp` has been normalized.
Computation of the normalization constant is automatic/implicit when fitting the distribution to a parametric one (e.g. the Gaussian one).
Otherwise, we usually delay its computation until strictly necessary
(for example, not during intermediate stages of conditioning, but at the end).
Note that when $ p(x|y)$ is not known
(has not been evaluated) for an entire grid of $x$,
but only in a few points,
then "how to normalize" becomes an important question too.
-->
''']

answers['pdf_U1'] = ['MD',r'''
    def pdf_U1(x,b,B):
        # Univariate (scalar), Uniform pdf

        pdf_values = ones((x-b).shape)

        a = b - sqrt(3*B)
        b = b + sqrt(3*B)

        pdf_values[x<a] = 0
        pdf_values[x>b] = 0

        height = 1/(b-a)
        pdf_values *= height

        return pdf_values
''']

answers['BR U1'] = ['MD',r'''
 - Because of the discretization.
 - The problem (of computing the posterior) is ill-posed:  
   The prior says there's zero probability of the truth being 
   in the region where the likelihood is not zero.
''']

answers['Dimensionality a'] = ['MD',r'''
$N^M$
''']
answers['Dimensionality b'] = ['MD',r'''
$15 * 360 * 180 = 972'000 \approx 10^6$
''']
answers['Dimensionality c'] = ['MD',r'''
$10^{10^6}$
''']

answers['BR Gauss'] = ['MD',r'''
We can ignore factors that do not depend on $x$.

\begin{align}
p(x|y)
&= \frac{p(x) \, p(y|x)}{p(y)} \\\
&\propto p(x) \, p(y|x) \\\
&=       N(x \mid b,B) \, N(y \mid x,R) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (x-b)^2/B + (x-y)^2/R \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (1/B + 1/R)x^2 - 2(b/B + y/R)x \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( x - \frac{b/B + y/R}{1/B + 1/R} \Big)^2 \cdot (1/B + 1/R) \Big) \, .
\end{align}

Identifying the last line with $N(x \mid \hat{x}, P)$ yields eqns (5) and (6).
''']

answers['KG intuition'] = ['MD',r'''
Because it describes how much the esimate is dragged from $b$ "towards" $y$.  
I.e. it is a multiplification (amplification) factor,
which French (signal processing) people like to call "gain".  

Relatedly, note that $K$ weights the observation uncertainty $(R)$ vs. the total uncertainty $(B+R)$,
and so is always between 0 and 1.
''']

answers['BR Gauss code'] = ['MD',r'''
    P  = 1/(1/B+1/R)
    mu = P*(b/B+y/R)
    # Gain version:
    #     KG = B/(B+R)
    #     P  = (1-KG)*B
    #     mu = b + KG*(y-b)
''']

answers['Posterior cov'] =  ['MD',r"""
  * No.
      * It means that information is always gained.
      * No, not always.  
        But on average, yes:  
        [the "expected" posterior entropy (and variance)
        is always smaller than that of the prior.](https://www.quora.com/What-conditions-guarantee-that-the-posterior-variance-will-be-less-than-the-prior-variance#)
  * It probably won't have decreased. Maybe you will just discard the new information entirely, in which case your certainty will remain the same.  
    I.e. humans are capable of thinking hierarchically, which effectively leads to other distributions than the Gaussian one.
"""]

answers['Why Gaussian'] =  ['MD',r"""
 * Simplicity: (recursively) yields "linear least-squares problems", whose solution is given by a linear systems of equations.
   This was demonstrated by the simplicity of the parametric Gaussian-Gaussian Bayes' rule.
 * The central limit theorem (CLT) and all its implications about likely noise distributions.
 * The intuitive precondition "ML estimator = sample average" necessitates a Gaussian sampling distribution.
 * For more, see chapter 7 of: [Probability theory: the logic of science](https://books.google.com/books/about/Probability_Theory.html?id=tTN4HuUNXjgC) (Edwin T. Jaynes), which is an excellent book for understanding probability and statistics.
"""]


###########################################
# Tut: Univariate Kalman filtering
###########################################

# Also see 'Gaussian sampling a'
answers['Gaussian sums'] = ['MD',r'''
By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
the mean parameter becomes:
$$ E(Fx+q) =  F E(x) + E(q) = F \hat{x} + \hat{q} \, . $$

Moreover, by independence,
$ Var(Fx+q) = Var(Fx) + Var(q) $,
and so
the variance parameter becomes:
$$ Var(Fx+q) = F^2 P + Q \, .  $$
''']

answers['LinReg deriv'] = ['MD',r'''
$$ \frac{d J_K}{d \hat{a}} = 0 = \ldots $$
''']

answers['LinReg_k'] = ['MD',r'''
    kk = 1+arange(k)
    a = sum(kk*yy[kk]) / sum(kk**2)
''']

answers['Sequential 2 Recusive'] = ['MD',r'''
    (k+1)/k
''']

answers['LinReg ⊂ KF'] = ['MD',r'''
The linear regression problem is formulated with $F_k = (k+1)/k$.  
The KF accepts a general $F_k$.
''']

answers['KF_k'] = ['MD',r'''
    ...
        else:
            PPf[k] = F(k-1)*PPa[k-1]*F(k-1) + Q
            xxf[k] = F(k-1)*xxa[k-1]
        # Analysis
        PPa[k] = 1/(1/PPf[k] + 1/R)
        xxa[k] = PPa[k] * (xxf[k]/PPf[k] + yy[k]/R)
        # Kalman gain form:
        # KG     = PPf[k] / (PPf[k] + R)
        # PPa[k] = (1-KG)*PPf[k]
        # xxa[k] = xxf[k]+KG*(yy[k]-xxf[k])
''']

answers['LinReg compare'] = ['MD',r'''

Let $\hat{a}_K$ denote the linear regression estimates of the slope $a$
based on the observations $y_1,\ldots, y_K$.  
Let $\hat{x}_K$ denote the KF estimate of $\hat{x}_K$ based on the same set of obs.  
It can bee seen in the plot that
$ \hat{x}_K = K \hat{a}_K \, . $
''']

answers['x_KF == x_LinReg'] = ['MD',r'''
We'll proceed by induction.  

With $P_1^f = \infty$, we get $P_1 \;(\text{i.e.}\; P_1^a)\; = R$,
which initializes (13).  

Now, inserting (13) in (12) yields:

$$
\begin{align}
P_{K+1} &= 1\Big/\big(1/R + \textstyle (\frac{K}{K+1})^2 / P_K\big)
\\\
&= R\Big/\big(1 + \textstyle (\frac{K}{K+1})^2 \frac{\sum_{k=1}^K k^2}{K^2}\big)
\\\
&= R\Big/\big(1 + \textstyle \frac{\sum_{k=1}^K k^2}{(K+1)^2}\big)
\\\
&= R(K+1)^2\Big/\big((K+1)^2 + \sum_{k=1}^K k^2\big)
\\\
&= R(K+1)^2\Big/\sum_{k=1}^{K+1} k^2
\, ,
\end{align}
$$
which concludes the induction.

The proof for (b) is similar.
''']

answers['Asymptotic P when F>1'] = ['MD',r'''
The fixed point $P_\infty$ should satisfy
$P_\infty = 1/\big(1/R + 1/[F^2 P_\infty]\big)$.
This yields $P_\infty = R (1-1/F^2)$.  
Interestingly, this means that the asymptotic state uncertainty ($P$)
is directly proportional to the observation uncertainty ($R$).
''']

answers['Asymptotic P when F=1'] = ['MD',r'''
Since
$ P_k^{-1} = P_{k-1}^{-1} + R^{-1} \, , $
it follows that
$ P_k^{-1} = P_0^{-1} + k R^{-1} \, , $
and hence
$$ P_k = \frac{1}{1/P_0 + k/R} \xrightarrow[k \rightarrow \infty]{} 0 \, .
$$
''']

answers['Asymptotic P when F<1'] = ['MD',r'''
Note that $P_k^a < P_k^f$ for each $k$
(c.f. the Gaussian-Gaussian Bayes rule from tutorial 2.)
Thus,
$$
P_k^a < P_k^f = F^2 P_{k-1}^f
\xrightarrow[k \rightarrow \infty]{} 0 \, .
$$
''']

answers['KG fail'] = ['MD',r'''
Because `PPa[0]` is infinite.
And while the limit (as `PPf` goes to +infinity) of
`KG = PPf / (PPf + R)` is 1,
its numerical evaluation fails (as it should).
Note that the infinity did not cause any problems numerically
for the "weighted average" form.
''']


###########################################
# Tut: Time series analysis
###########################################


###########################################
# Tut: Multivariate Kalman
###########################################

answers['Likelihood derivation'] = ['MD',r'''
$$
'''+macros+r'''
$$
Imagine that $\y=\br$ (instead of eqn 2),
then the distribution of $\y$ would be the same as for $\br$.
The only difference is that we've added $\bH \x$, which is a (deterministic/fixed) constant, given $\x$.
Adding a constant to a random variable just changes its mean,
hence $\mathcal{N}(\y \mid \bH \x, \R)$

A more formal (but not really more rigorous) explanation is as follows:
$$
\begin{align}
p(\y|\x)
&= \int p(\y,\br|\x) \, d \br \tag{by law of total proba.}  \\\
&= \int p(\y|\br,\x) \, p(\br|\x) \, d \br \tag{by def. of conditional proba.} \\\
&= \int \delta\big(\y-(\bH \x + \br)\big) \, p(\br|\x) \, d \br \tag{$\y$ is fully determined by $\x$ and $\br$} \\\
&= \int \delta\big(\y-(\bH \x + \br)\big) \, \mathcal{N}(\br \mid 0, \R) \, d \br \tag{the draw of $\br$ does not depened on $\x$} \\\
&= \mathcal{N}(\y - \bH \x \mid 0, \R) \tag{by def. of Dirac Delta} \\\
&= \mathcal{N}(\y \mid \bH \x, \R) \tag{by reformulation} \, .
\end{align}
$$
''']


answers['KF precision'] = ['MD',r'''
By Bayes' rule:
$$
'''+macros+r'''
\begin{align}
- 2 \log p(\x|\y) =
\norm{\bH \x-\y}\_\R^2 + \norm{\x - \bb}\_\B^2
 + \cx_1
\, .
\end{align}
$$
Expanding, and gathering terms of equal powers in $\x$ yields:
$$
\begin{align}
- 2 \log p(\x|\y)
&=
\x\tr \left( \bH\tr \Ri \bH + \Bi  \right)\x
- 2\x\tr \left[\bH\tr \Ri \y + \Bi \bb\right] + \cx_2
\, .
\end{align}
$$
Meanwhile
$$
\begin{align}
\norm{\x-\hat{\x}}_\bP^2
&=
\x\tr \bP^{-1} \x - 2 \x\tr \bP^{-1} \hat{\x} + \cx_3
\, .
\end{align}
$$
Eqns (5) and (6) follow by identification.
''']


# Also comment on CFL condition (when resolution is increased)?
answers['Cov memory'] = ['MD',r'''
 * (a). $M$-by-$M$
 * (b). Using the [cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation),
    at least 2 times $M^3/3$.
 * (c). Assume $\mathbf{B}$ stored as float (double). Then it's 8 bytes/element.
 And the number of elements in $\mathbf{B}$: $M^2$. So the total memory is $8 M^2$.
 * (d). 8 trillion bytes. I.e. 8 million MB. 
''']


answers['Woodbury'] = ['MD',r'''
We show that they cancel:
$$
'''+macros+r'''
\begin{align}
  &\left(\B^{-1}+\V\tr \R^{-1} \U \right)
  \left[ \B - \B \V\tr \left(\R+\U \B \V\tr \right)^{-1} \U \B \right] \\\
  & \quad = \I_M + \V\tr \R^{-1}\U \B -
  (\V\tr + \V\tr \R^{-1} \U \B \V\tr)(\R + \U \B \V\tr)^{-1}\U \B \\\
  & \quad = \I_M + \V\tr \R^{-1}\U \B -
  \V\tr \R^{-1}(\R+ \U \B \V\tr)(\R + \U \B \V\tr)^{-1} \U \B \\\
  & \quad = \I_M + \V\tr \R^{-1} \U \B - \V\tr \R^{-1} \U \B \\\
  & \quad = \I_M
\end{align}
$$
''']

answers['Woodbury C1'] = ['MD',r'''
$
'''+macros+r'''
$The corollary follows from the Woodbury identity
by replacing $\V,\U$ by $\bH$,
*provided that everything is still well-defined*.
In other words,
we need to show the existence of the left hand side.

Now, for all $\x \in \Reals^M$, $\x\tr \B^{-1} \x > 0$ (since $\B$ is SPD).
Similarly, $\x\tr \bH\tr \R^{-1} \bH \x\geq 0$,
implying that the left hand side is SPD:
$\x\tr (\bH\tr \R^{-1} \bH + \B^{-1})\x > 0$,
and hence invertible.
''']

answers['Woodbury C2'] = ['MD',r'''
$
'''+macros+r'''
$A straightforward validation of (C2)
is obtained by cancelling out one side with the other.
A more satisfying exercise is to derive it from (C1)
starting by right-multiplying by $\bH\tr$.
''']


###########################################
# Tut: Dynamical systems, chaos, Lorenz
###########################################

answers["Ergodicity a"] = ["MD",r'''
For asymptotically large $T$, the answer is "yes";
however, this is difficult to distinguish if $T<60$ or $N<400$,
which takes a very long time with the integrator used in the above.
''']

answers["Ergodicity b"] = ["MD",r'''
It doesn't matter
(provided the initial conditions for each experiment is "cropped out" before averaging).
''']

answers["Hint: Lorenz energy"] = ["MD",r'''
Hint: what's its time-derivative?
''']

answers["Lorenz energy"] = ["MD",r'''
\begin{align}
\frac{d}{dt}
\sum_m
x_m^2
&=
2 \sum_m
x_m \dot{x}_m
\end{align}

Next, insert the quadratic terms from the ODE,
$
\dot x_m = (x_{m+1} − x_{m-2}) x_{m-1}
\, .
$

Finally, apply the periodicity of the indices.
''']

answers["error evolution"] = ["MD",r"""
$\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
"""]
answers["anti-deriv"] = ["MD",r"""
Differentiate $e^{F t}$.
"""]
answers["predictability cases"] = ["MD",r"""
* (1). Dissipates to 0.
* (2). No.
      A balance is always reached between
      the uncertainty reduction $(1-K)$ and growth $F^2$.  
      Also recall the asymptotic value of $P_k$ computed in
      [T3](T3 - Univariate Kalman filtering.ipynb#Exc-3.14:).
"""]
answers["saturation term"] = ["MD",r"""
[link](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
"""]
answers["liner growth"] = ["MD",r"""
$\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
"""]

answers["doubling time"] = ["MD",r"""
    xx   = output_63[0][:,-1]      # Ensemble of particles at the end of integration
    v    = np.var(xx, axis=0)      # Variance (spread^2) of final ensemble
    v    = mean(v)                 # homogenize
    d    = sqrt(v)                 # std. dev.
    eps  = [FILL IN SLIDER VALUE]  # initial spread
    T    = [FILL IN SLIDER VALUE]  # integration time
    rate = log(d/eps)/T            # assuming d = eps*exp(rate*T)
    print("Doubling time (approx):",log(2)/rate)
"""]


###########################################
# Tut: Ensemble [Monte-Carlo] approach
###########################################

# answers['Gaussian sampling a'] = ['MD',r'''
# Firstly, a linear (affine) transformation can be decomposed into a sequence of sums. This means that $\mathbf{x}$ will be Gaussian.
# It remains only to calculate its moments.

# By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
# $$E(\mathbf{x}) = E(\mathbf{L} \mathbf{z} + \mathbf{b}) = \mathbf{L} E(\mathbf{z}) + \mathbf{b} = \mathbf{b} \, .$$

# Moreover,
# $$\newcommand{\b}{\mathbf{b}} \newcommand{\x}{\mathbf{x}} \newcommand{\z}{\mathbf{z}} \newcommand{\L}{\mathbf{L}}
# E((\x - \b)(\x - \b)^T) = E((\L \z)(\L \z)^T) = \L E(\z^{} \z^T) \L^T = \L \mathbf{I}_m \L^T = \L \L^T \, .$$
# ''']

answers['KDE'] = ['MD',r'''
    from scipy.stats import gaussian_kde`
    ax.plot(xx,gaussian_kde(E.ravel()).evaluate(xx),label="KDE estimate")
''']

answers['Gaussian sampling a'] = ['MD',r'''

Type `randn??` in a code cell and execute it.
''']
answers['Gaussian sampling b'] = ['MD',r'''
    z = randn((M,1))
    x = b + L @ z
''']

answers['Gaussian sampling c'] = ['MD',r'''
    E = b[:,None] + L @ randn((M,N))
    # Alternatives:
    # E = np.random.multivariate_normal(b,B,N).T
    # E = ( b + randn((N,M)) @ L.T ).T
''']

answers['Average sampling error'] = ['MD',r'''
Procedure:

 1. Repeat the experiment many times.
 2. Compute the average error ("bias") of $\overline{\mathbf{x}}$. Verify that it converges to 0 as $N$ is increased.
 3. Compute the average *squared* error. Verify that it is approximately $\text{diag}(\mathbf{B})/N$.
''']

answers['ensemble moments'] = ['MD',r'''
    x_bar = np.sum(E,axis=1)/N
    B_bar = zeros((M,M))
    for n in range(N):
        xc = (E[:,n] - x_bar)[:,None] # x_centered
        B_bar += xc @ xc.T
        #B_bar += np.outer(xc,xc)
    B_bar /= (N-1)
''']

answers['Why (N-1)'] = ['MD',r'''
 * [Unbiased](https://en.wikipedia.org/wiki/Variance#Sample_variance)
 * Suppose we compute the square root of this estimate. Is this an unbiased estimator for the standard deviation?
''']

answers['ensemble moments vectorized'] = ['MD',r'''
 * (a). Show that element $(i,j)$ of the matrix product $\mathbf{X}^{} \mathbf{Y}^T$
 equals element $(i,j)$ of the sum of the outer product of their columns:
 $\sum_n \mathbf{x}_n \mathbf{y}_n^T$.
 Put this in the context of $\overline{\mathbf{B}}$.
 * (b). Use the following
 
code:

    x_bar = np.sum(E,axis=1,keepdims=True)/N
    X     = E - x_bar
    B_bar = X @ X.T / (N-1)   
''']

# Skipped
answers['Why matrix notation'] = ['MD',r'''
   - Removes indices
   - Highlights the linear nature of many computations.
   - Tells us immediately if we're working in state space or ensemble space
     (i.e. if we're manipulating individual dimensions, or ensemble members).
   - Helps with understanding subspace rank issues
   - Highlights how we work with the entire ensemble, and not individual members.
   - Suggest a deterministic parameterization of the distributions.
''']

answers['estimate cross'] = ['MD',r'''
    def estimate_cross_cov(Ex,Ey):
        N = Ex.shape[1]
        assert N==Ey.shape[1]
        X = Ex - np.mean(Ex,axis=1,keepdims=True)
        Y = Ey - np.mean(Ey,axis=1,keepdims=True)
        CC = X @ Y.T / (N-1)
        return CC
''']

answers['errors'] = ['MD',r'''
 * (a). Error: discrepancy from estimator to the parameter targeted.
Residual: discrepancy from explained to observed data.
 * (b). Bias = *average* (i.e. systematic) error.
 * (c). [Wiki](https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship)
''']


###########################################
# Tut: Writing your own EnKF
###########################################
answers['EnKF v1'] = ['MD',r'''
    def my_EnKF(N):
        E = mu0[:,None] + P0_chol @ randn((M,N))
        for k in range(1,K+1):
            # Forecast
            t   = k*dt
            E   = f(E,t-dt,dt)
            E  += Q_chol @ randn((M,N))
            if not k%dkObs:
                # Analysis
                y        = yy[k//dkObs-1] # current obs
                Eo       = Obs(E,t)
                PH       = estimate_cross_cov(E,Eo)
                HPH      = estimate_mean_and_cov(Eo)[1]
                Perturb  = R_chol @ randn((p,N))
                KG       = divide_1st_by_2nd(PH, HPH+R)
                E       += KG @ (y[:,None] - Perturb - Eo)
            mu[k] = mean(E,axis=1)
''']

answers['rmse'] = ['MD',r'''
    rmses = sqrt(np.mean((xx-mu)**2, axis=1))
    average = np.mean(rmses)
''']

answers['Repeat experiment a'] = ['MD',r'''
 * (a). Set `p=1` above, and execute all cells below again.
''']

answers['Repeat experiment b'] = ['MD',r'''
 * (b). Insert `seed(i)` for some number `i` above the call to the EnKF or above the generation of the synthetic truth and obs.
''']

answers['Repeat experiment cd'] = ['MD',r'''
 * (c). Void.
 * (d). Use: `Perturb  = D_infl * R_chol @ randn((p,N))` in the EnKF algorithm.
''']


###########################################
# Tut: Benchmarking with DAPPER
###########################################
answers['jagged diagnostics'] = ['MD',r'''
Because they are only defined at analysis times, i.e. every `dkObs` time step.
''']

answers['RMSE hist'] = ['MD',r'''
 * The MSE will be (something close to) chi-square.
 * That the estimator and truth are independent, Gaussian random variables.
''']

answers['Rank hist'] = ['MD',r'''
 * U-shaped: Too confident
 * A-shaped: Too uncertain
 * Flat: well calibrated
''']

# Pointless...
# Have a look at the phase space trajectory output from `plot_3D_trajectory` above.
# The "butterfly" is contained within a certain box (limits for $x$, $y$ and $z$).
answers['RMSE vs inf error'] = ['MD',r'''
It follows from [the fact that](https://en.wikipedia.org/wiki/Lp_space#Relations_between_p-norms)
$ \newcommand{\x}{\mathbf{x}} \|\x\|_2 \leq M^{1/2} \|\x\|\_\infty \text{and}  \|\x\|_1 \leq M^{1/2} \|\x\|_2$
that
$$ 
\text{RMSE} 
= \frac{1}{K}\sum_k \text{RMSE}_k
\leq \| \text{RMSE}\_{0:k} \|\_\infty
$$
and
$$ \text{RMSE}_k = \| \text{Error}_k \|\_2 / \sqrt{M} \leq \| \text{Error}_k \|\_\infty$$
''']

answers['Twin Climatology'] = ['MD',r'''
    config = Climatology(**defaults)
    avergs = config.assimilate(HMM,xx,yy).average_in_time()
    print_averages(config,avergs,[],['rmse_a','rmv_a'])
''']

answers['Twin Var3D'] = ['MD',r'''
    config = Var3D(**defaults)
    ...
''']


###########################################
# Colin Grudzen
###########################################

answers['forward_euler'] = ['MD', r'''
Missing line:

    xyz_step = xyz + dxdt(xyz, h, sigma=SIGMA, beta=BETA, rho=RHO) * h
''']

answers['log_growth'] = ['MD', r'''
Missing lines:

    nrm = sqrt( (x_pert_k - x_control_k) @ (x_pert_k - x_control_k).T )

    log_growth_rate = (1.0 / T) * log(nrm / eps)
''']


answers['power_method'] = ['MD', r'''
Missing lines:

        v = M @ v
        v = v / sqrt(v.T @ v)
    
    mu = v.T @ M @ v
''']

answers['power_method_convergence_rate'] = ['HTML', r'''
Suppose we have a random vector <span style="font-size:1.25em">$\mathbf{v}_0$</span>.  If <span style="font-size:1.25em">$\mathbf{M}$</span> is diagonalizable, then we can write <span style="font-size:1.25em">$\mathbf{v}_0$</span> in a basis of eigenvectors, i.e.,
<h3>$$v_0 = \sum_{j=1}^n \alpha_j \nu_j,$$ </h3>
where  <span style="font-size:1.25em">$\nu_j$</span> is an eigenvector for the eigenvalue  <span style="font-size:1.25em">$\mu_j$</span>, and  <span style="font-size:1.25em">$\alpha_j$</span> is some coefficient in  <span style="font-size:1.25em">$\mathbb{R}$</span>.  We consider thus, with probability one,  <span style="font-size:1.25em">$\alpha_1 \neq 0$</span>. 


In this case, we note that
<h3>
$$\mathbf{M}^k \mathbf{v}_0 = \mu_1^k  \left( \alpha_1 \nu_1 + \sum_{j=2}^n \alpha_j \left(\frac{\mu_j}{\mu_1}\right)^k \nu_j\right).
$$</h3>

But 
<h3>$$\frac{\rvert \mu_j\rvert}{\rvert\mu_1\rvert} <1$$</h3>
for each  <span style="font-size:1.25em">$j>1$</span>, so that the projection of  <span style="font-size:1.25em">$\mathbf{M}^k \mathbf{v}_0$</span> into each eigenvector  <span style="font-size:1.25em">$\{\nu_j\}_{j=2}^n$</span> goes to zero at a rate of at least  
<h3>
$$\mathcal{O}  \left(\left[ \frac{\lambda_2}{\lambda_1} \right]^k \right).$$
</h3>
We need only note that  <span style="font-size:1.25em">$\mathbf{M}^k \mathbf{v}_0$</span> and  <span style="font-size:1.25em">$\mathbf{v}_{k}$</span> share the same span.
''']


answers['lyapunov_exp_power_method'] = ['HTML', r'''
<ol>
<li>Consider, if  <span style="font-size:1.25em">$ \widehat{\mu}_k \rightarrow \mu_1$</span>  as  <span style="font-size:1.25em">$k \rightarrow \infty$</span>, then for all  <span style="font-size:1.25em">$\epsilon>0$</span> there exists a  <span style="font-size:1.25em">$T_0$</span> such that,<h3>$ \rvert \mu_1 \rvert - \epsilon < \rvert \widehat{\mu}_k\rvert < \rvert \mu_1 \rvert + \epsilon $, </h3>
<br>
for all <span style="font-size:1.25em">$k > T_0$</span>.  In particular, we will choose some  <span style="font-size:1.25em">$\epsilon$ </span> sufficiently small such that,
<h3>$$\begin{align}
\rvert \mu_1 \rvert - \epsilon > 0.
\end{align}$$</h3>
<br>
This is possible by the assumption <span style="font-size:1.25em">$\rvert \mu_1 \rvert >0$</span>.

We will write,
<h3>$\widehat{\lambda}_T =\frac{1}{T} \sum_{k=1}^{T_0} \log\left(\rvert \widehat{\mu}_1 \rvert\right) + \frac{1}{T} \sum_{k=T_0 +1}^T  \log \left(\rvert \widehat{\mu}_1 \rvert\right)$. </h3>
<br>
We note that  <span style="font-size:1.25em">$\log$</span> is monotonic, so that for  <span style="font-size:1.25em">$T> T_0$</span>,

<h3>$\frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \mu_1\rvert - \epsilon \right) < \frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert\widehat{\mu}_k \rvert \right) <\frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \mu_1\rvert + \epsilon \right)$.</h3>
<br>
But that means,

<h3>$\frac{T - T_0}{T} \log\left(\rvert \mu_1\rvert - \epsilon \right) < \frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \widehat{\mu}_k \rvert \right) <\frac{T - T_0}{T}  \log\left(\rvert \mu_1\rvert  + \epsilon \right)$.</h3>
<br>
Notice that in limit,

<h3>$\lim_{T\rightarrow \infty}\frac{1}{T}\sum_{k=1}^{T_0} \log\left(\rvert \widehat{\mu}_k\rvert \right) = 0$, </h3>
<br>
and therefore we can show,

<h3>$\log\left(\rvert \mu_1 \rvert - \epsilon \right) < \lim_{T \rightarrow \infty} \widehat{\lambda}_T < \log\left(\rvert \mu_1 \rvert + \epsilon \right),$</h3>
<br>
for all  <span style="font-size:1.25em">$\epsilon >0$</span>.  This shows that

<h3>$ \lim_{T \rightarrow \infty} \widehat{\lambda}_T = \log\left(\rvert \mu_1\rvert \right). $</h3></li>
<br>
<li>
The Lyapunov exponents for the fixed matrix  <span style="font-size:1.25em">$\mathbf{M}$</span> are determined by the log, absolute value of the eigenvalues.
</li>
</ol>
''']


answers['fixed_point'] = ['HTML', r'''
Suppose for all components  <span style="font-size:1.25em">$x^j$</span> we choose  <span style="font-size:1.25em">$x^j = F$</span>.  The time derivative at this point is clearly zero.
''']

answers['probability_one'] = ['HTML', r'''
We relied on the fact that there is probability one that a Gaussian distributed vector has a nonzero projection into the eigenspace for the leading eigenvalue.  Consider why this is true.

Let  <span style="font-size:1.25em">$\{\mathbf{v}_j \}_{j=1}^n $</span> be any orthonormal basis such that  <span style="font-size:1.25em">$\mathbf{v}_1$</span> is an eigenvector for  <span style="font-size:1.25em">$\mu_1$</span>.  Let 
<h3>$$
\chi(\mathbf{x}) : \mathbb{R}^n \rightarrow \{0, 1\}
$$</h3>
<br>
be the indicator function on the span of  <span style="font-size:1.25em">$\{\mathbf{v}_j\}_{j=2}^n$</span>, i.e., the hyper-plane orthogonal to  <span style="font-size:1.25em">$\mathbf{v}_1$</span>.  The probability of choosing a Gaussian distributed random vector that has no component in the span of <span style="font-size:1.25em">$\mathbf{v}_1$</span> is measured by integrating
<h2>$$
\frac{1}{\left(2\pi\right)^n}\int_{\mathbb{R}} \cdots \int_{\mathbb{R}}\chi\left(\sum_{j=1}^n  \alpha_j v_j \right)
 e^{\frac{-1}{2} \sum_{j=1}^n \alpha_j^2 }
{\rm d}\alpha_1  \cdots {\rm d} \alpha_n.
$$</h2>
<br>
But  <span style="font-size:1.25em">$\chi \equiv 0$</span> whenever  <span style="font-size:1.25em">$\alpha_1 \neq 0$</span>, and  <span style="font-size:1.25em">${\rm d} \alpha_1 \equiv 0$</span> on this set.  This means that the probability of selecting a Gaussian distributed vector with  <span style="font-size:1.25em">$\alpha_1 =0$</span> is equal to zero.
<br>
In more theoretical terms, this corresponds to the hyper-plane having measure zero with respect to the Lebesgue measure.
''']

answers['gram-schmidt'] = ['HTML', r'''
The vectors, <span style="font-size:1.25em">$\{\mathbf{x}_0^1, \mathbf{x}_0^2 \}$</span> are related to the vectors <span style="font-size:1.25em">$\{\mathbf{x}_1^1, \mathbf{x}_1^2 \}$</span> by propagating forward via the matrix <span style="font-size:1.25em">$\mathbf{M}$</span>, and the Gram-Schmidt step.  Thus by writing,
<h3>$$\begin{align}
\widehat{\mathbf{x}}_1^2 &\triangleq \mathbf{y}^2_1  + \langle \mathbf{x}_1^1,  \widehat{\mathbf{x}}^2_1\rangle \mathbf{x}_1^1
\end{align}$$</h3>
<br>
it is easy to see
<h3>$$\begin{align}
\mathbf{M} \mathbf{x}_0^1 &= U^{11}_1 \mathbf{x}_1^1 \\
\mathbf{M} \mathbf{x}_0^2 &= U^{22}_1 \mathbf{x}_1^2 + U^{12}_1 \mathbf{x}_1^1.
\end{align}$$</h3>
<br>
This leads naturally to an upper triangular matrix recursion.  Define the following matrices, for <span style="font-size:1.25em">$k \in \{1,2, \cdots\}$</span>
<h3>$$\begin{align}
\mathbf{U}_k \triangleq \begin{pmatrix}
U_k^{11} & U_k^{12} \\
0 & U_k^{22}
\end{pmatrix} & & \mathbf{E}_{k-1} \triangleq \begin{pmatrix}
\mathbf{x}_{k-1}^{1} & \mathbf{x}_{k-1}^{2},
\end{pmatrix}
\end{align}$$</h3>
<br>
then in matrix form, we can write the recursion for an arbitrary step $k$ as
<h3>$$\begin{align}
\mathbf{M} \mathbf{E}_k = \mathbf{E}_{k+1} \mathbf{U}_{k+1}
\end{align}$$</h3>
<br>
where the coefficients of <span style="font-size:1.25em">$\mathbf{U}_k$</span> are defined by the Gram-Schmidt step. described above.
''']

answers['schur_decomposition'] = ['HTML', r'''
We can compute the eigenvalues as the roots of the characteristic polynomial.  Specifically, the characteristic polynomial is equal to
<h3>$$\begin{align}
\det\left( \mathbf{M} - \lambda \mathbf{I} \right) &= \det\left( \mathbf{Q} \mathbf{U} \mathbf{Q}^{\rm T} - \lambda\mathbf{I} \right) \\
&=\det\left( \mathbf{Q}\left[ \mathbf{U}   - \lambda\mathbf{I}\right] \mathbf{Q}^{\rm T} \right) \\
&=\det\left( \mathbf{Q}\right) \det\left( \mathbf{U}   - \lambda\mathbf{I}\right) \det\left(\mathbf{Q}^{\rm T} \right) \\
&=\det\left( \mathbf{Q} \mathbf{Q}^{\rm T} \right) \det\left( \mathbf{U}   - \lambda\mathbf{I}\right) \\
&=\det\left( \mathbf{U}   - \lambda\mathbf{I}\right)
\end{align}$$ </h3>
<br>
By expanding the determinant in co-factors, it is easy to show that the determinant of the right hand side equals
<h3>$$\begin{align}
\prod_{j=1}^n (U^{jj} - \lambda).
\end{align}$$ </h3>
<br>

By orthogonality, it is easy to verify that 
<h3>$$\begin{align}
\left(\mathbf{Q}^j\right)^{\rm T} \mathbf{M} \mathbf{Q}^j = U^{jj}.
\end{align}$$</h3>

''']

answers['lyapunov_vs_es'] = ['HTML', r'''
We define the <b><em>i</em>-th Lyapunov exponent</b> as
<h3>$$\begin{align}
\lambda_i & \triangleq \lim_{k\rightarrow \infty} \frac{1}{k}\sum_{j=1}^k \log\left(\left\rvert U_j^{ii}\right\rvert \right)
\end{align}$$</h3>
<br>
and the <b><em>i</em>-th (backward) Lyapunov vector at time <em>k</em></b> to be the $i$-th column of <span style="font-size:1.25em"> $\mathbf{E}_k$ </span>.
''']

answers['naive_QR'] = ['MD', r'''
Example solution:

        perts[:, i] = perts[:, i] - sqrt(perts[:, i].T @ perts[:, j]) perts[:, j]
    
    perts[:, i] = perts[:, i] / sqrt(perts[:,i].T @ perts[:, i])
''']

answers['real_schur'] = ['HTML', r'''
Let <span style='font-size:1.25em'>$\mathbf{M}$</span> be any matrix in <span style='font-size:1.25em'>$\mathbb{R}^{n\times n}$</span> with eigenvalues ordered
<h3>
$$
\begin{align}
\rvert \mu_1 \rvert \geq \cdots \geq \rvert \mu_s \rvert.
\end{align}
$$
</h3>
<br>
A real Schur decomposition of <span style='font-size:1.25em'>$\mathbf{M}$</span> is defined via 
<h3>
$$
\begin{align}
\mathbf{M} = \mathbf{Q} \mathbf{U} \mathbf{Q}^{\rm T}
\end{align}
$$
</h3>
<br>
where <span style='font-size:1.25em'>$\mathbf{Q}$</span> is an orthogonal matrix and <span style='font-size:1.25em'>$\mathbf{U}$</span> is a block upper triangular matrix, such that
<h3>$$
\begin{align}
\mathbf{U} \triangleq
\begin{pmatrix}
U^{11} & U^{12} & \cdots & U^{1n} \\
 0 & U^{22} & \cdots & U^{2n} \\
 \vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & U^{nn}
\end{pmatrix}.
\end{align}
$$
</h3>
<br>
Moreover, the eigenvalues of <span style='font-size:1.25em'>$\mathbf{U}$</span> must equal the eigenvalues of <span style='font-size:1.25em'>$\mathbf{M}$</span>, such that: 
<ol>
<li> each diagonal block <span style='font-size:1.25em'>$U^{ii}$</span> is either a scalar or a $2\times 2$ matrix with complex conjugate eigenvalues, and </li>
<li> the eigenvalues of the diagonal blocks <span style='font-size:1.25em'>$U^{ii}$</span> are ordered descending in magnitude.
</ol>
''']                     


###########################################
# Topic
###########################################




