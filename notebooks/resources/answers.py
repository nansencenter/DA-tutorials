from markdown import markdown as md2html # better than markdown2 ?
from IPython.display import HTML, display

from .macros import include_macros

def show_answer (tag): formatted_display(*answers[tag], '#dbf9ec') #d8e7ff
def show_example(tag): formatted_display(*answers[tag], '#ffed90')

def formatted_display(TYPE,content,bg_color):

    # Remove 1st linebreak
    content = content[1:]

    # Convert from TYPE to HTML
    if   TYPE == "HTML": content = content
    elif TYPE == "TXT" : content = '<pre><code>'+content+'</code></pre>'
    elif TYPE == "MD"  : content = md2html(include_macros(content))

    # Make bg style
    bg_color = 'background-color:'+ bg_color + ';' #d8e7ff #e2edff

    # Compose string
    content = '<div style="'+bg_color+'padding:0.5em;">'+str(content)+'</div>'

    # Fix Colab - MathJax incompatibility
    setup_typeset()

    # Display
    display(HTML(content))

def setup_typeset():
    """MathJax initialization for the current cell.

    This installs and configures MathJax for the current output.

    Necessary in Google Colab. Ref:
    https://github.com/googlecolab/colabtools/issues/322
    """

    # Only run in Colab
    try:
        import google.colab
    except ImportError:
        return

    # Note: The original function enabled \( math \) and \[ math \] style, with:
    #    'inlineMath': [['$', '$'], ['\\(', '\\)']],
    #    'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    # but I disabled this coz regular Jupyter does not support this,
    # and it breaks MD rendering of regular parantheses and brackets.


    URL = '''https://colab.research.google.com/static/mathjax/MathJax.js'''
    URL += '''?config=TeX-AMS_HTML-full,Safe&delayStartupUntil=configured'''
    script1 = '''<script src="%s"></script>'''%URL
    # Alternative URL (also need config?):
    # - https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js 
    # - https://www.gstatic.com/external_hosted/mathjax/latest/MathJax.js

    display(HTML(script1 + '''
            <script>
                (() => {
                    const mathjax = window.MathJax;
                    mathjax.Hub.Config({
                    'tex2jax': {
                        'inlineMath': [['$', '$']],
                        'displayMath': [['$$', '$$']],
                        'processEscapes': true,
                        'processEnvironments': true,
                        'skipTags': ['script', 'noscript', 'style', 'textarea', 'code'],
                        'displayAlign': 'center',
                    },
                    'HTML-CSS': {
                        'styles': {'.MathJax_Display': {'margin': 0}},
                        'linebreaks': {'automatic': true},
                        // Disable to prevent OTF font loading, which aren't part of our
                        // distribution.
                        'imageFont': null,
                    },
                    'messageStyle': 'none'
                });
                mathjax.Hub.Configured();
            })();
            </script>
            '''))

answers = {}

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

answers['Discussion topics 1'] = ['MD',r'''
 * (a). State estimation for large systems.
 * (b). States are (unknown) variables that change in time.  
 For a given dynamical system, the chosen parameterisation  
 should contain all prognostic variables, and be fairly non-redundant.
 * (c). Stuff that changes in time.
 * (d). In principle it's a science. In practice...
 * (e). Abstract concept to break the problem down into smaller, recursive, problems.  
DAGs. Formalises the concept of hidden variables (states).
''']

###########################################
# Tut: Bayesian inference
###########################################
answers['pdf_G1'] = ['MD',r'''
    pdf_values = 1/sqrt(2*pi*B)*exp(-0.5*(x-b)**2/B)
    # Version using the scipy (sp) library:
    # pdf_values = sp.stats.norm.pdf(x,loc=b,scale=sqrt(B))
''']

answers['BR'] = ['MD',r'''
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
Firstly, note that normalization is quite necessary, being requried by any expectation computation. For example, $\Expect(x|y) = \int x \, p(x) \, dx \approx$ `x*pp*dx` is only valid if `pp` has been normalized.
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
    P    = 1/(1/B+1/R)
    xhat = P*(b/B+y/R)
    # Gain version:
    #     KG   = B/(B+R)
    #     P    = (1-KG)*B
    #     xhat = b + KG*(y-b)
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
answers['RV sums'] = ['MD',r'''
By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
and that of (Dyn),
the mean parameter becomes:
$$ \\Expect(\\DynMod x+q) =  \\DynMod \\Expect(x) + \\Expect(q) = \\DynMod \hat{x} \, . $$

Moreover, by independence,
$ \\text{Var}(\\DynMod x+q) = \\text{Var}(\\DynMod x) + \\text{Var}(q) $,
and so
the variance parameter becomes:
$$ \\text{Var}(\\DynMod x+q) = \\DynMod^2 P + Q \, .  $$
''']

answers['LinReg deriv a'] = ['MD',r'''
$$ \begin{align}
p\, (y_1, \ldots, y_K \;|\; a)
&= \prod_{k=1}^K \, p\, (y_k \;|\; a) \tag{each obs. is indep. of others, knowing $a$.} \\\
&= \prod_k \, \mathcal{N}(y_k \mid a k,R) \tag{inserted eqn. (3) and then (1).} \\\
&= \prod_k \,  (2 \pi R)^{-1/2} e^{-(x - a k)^2/2 R} \tag{inserted eqn. (G1) from T2.} \\\
&= c \exp\Big(\frac{-1}{2 R}\sum_k (x - a k)^2\Big) \\\
\end{align} $$
Taking the logarithm is a monotonic transformation, so it does not change the location of the maximum location.
Neither does diving by $c$. Multiplying by $-2 R$ means that the maximum becomes the minimum.
''']

answers['LinReg deriv b'] = ['MD',r'''
$$ \frac{d J_K}{d \hat{a}} = 0 = \ldots $$
''']

answers['LinReg_k'] = ['MD',r'''
    kk = 1+arange(k)
    a = sum(kk*yy[kk]) / sum(kk**2)
''']

answers['Sequential 2 Recursive'] = ['MD',r'''
    (k+1)/k
''']

answers['LinReg ⊂ KF'] = ['MD',r'''
Linear regression is only optimal if the truth is a straight line,
i.e. if $\\DynMod_k = (k+1)/k$.  

Compared to the KF, which accepts a general $\\DynMod_k$,
this is so restrictive that one does not usually think
of the methods as belonging to the same class at all.
''']

answers['KF_k'] = ['MD',r'''
    ...
        else:
            BB[k] = Mod(k-1)*PP[k-1]*Mod(k-1) + Q
            bb[k] = Mod(k-1)*xxhat[k-1]
        # Analysis
        PP[k]    = 1/(1/BB[k] + 1/R)
        xxhat[k] = PP[k] * (bb[k]/BB[k] + yy[k]/R)
        # Kalman gain form:
        # KG       = BB[k] / (BB[k] + R)
        # PP[k]    = (1-KG)*BB[k]
        # xxhat[k] = bb[k]+KG*(yy[k]-bb[k])
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

With $B_1 = \infty$, we get $P_1 = R$,
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

The proof for $\hat{x}_k$ is similar.
''']

answers['Asymptotic P when M>1'] = ['MD',r'''
The fixed point $P_\infty$ should satisfy
$P_\infty = 1/\big(1/R + 1/[\\DynMod^2 P_\infty]\big)$.
This yields $P_\infty = R (1-1/\\DynMod^2)$.  
Interestingly, this means that the asymptotic state uncertainty ($P$)
is directly proportional to the observation uncertainty ($R$).
''']

answers['Asymptotic P when M=1'] = ['MD',r'''
Since
$ P_k^{-1} = P_{k-1}^{-1} + R^{-1} \, , $
it follows that
$ P_k^{-1} = P_0^{-1} + k R^{-1} \, , $
and hence
$$ P_k = \frac{1}{1/P_0 + k/R} \xrightarrow[k \rightarrow \infty]{} 0 \, .
$$
''']

answers['Asymptotic P when M<1'] = ['MD',r'''
Note that $P_k < B_k$ for each $k$
(c.f. the Gaussian-Gaussian Bayes rule from tutorial 2.)
Thus,
$$
P_k < B_k = \\DynMod^2 B_{k-1}
\xrightarrow[k \rightarrow \infty]{} 0 \, .
$$
''']

answers['KG fail'] = ['MD',r'''
Because `PP[0]` is infinite.
And while the limit (as `BB` goes to +infinity) of
`KG = BB / (BB + R)` is 1,
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
&= \int \delta\big(\y-(\bH \x + \br)\big) \, \mathcal{N}(\br \mid \\bvec{0}, \R) \, d \br \tag{the draw of $\br$ does not depened on $\x$} \\\
&= \mathcal{N}(\y - \bH \x \mid \\bvec{0}, \R) \tag{by def. of Dirac Delta} \\\
&= \mathcal{N}(\y \mid \bH \x, \R) \tag{by reformulation} \, .
\end{align}
$$
''']

answers['KF precision'] = ['MD',r'''
By Bayes' rule:
$$\begin{align}
- 2 \log p(\x|\y) =
\|\bH \x-\y \|\_\R^2 + \| \x - \bb \|\_\B^2
 + \text{const}_1
\, .
\end{align}$$
Expanding, and gathering terms of equal powers in $\x$ yields:
$$\begin{align}
- 2 \log p(\x|\y)
&=
\x\tr \left( \bH\tr \Ri \bH + \Bi  \right)\x
- 2\x\tr \left[\bH\tr \Ri \y + \Bi \bb\right] + \text{const}_2
\, .
\end{align}$$
Meanwhile
$$\begin{align}
\| \x-\hat{\x} \|\_\bP^2
&=
\x\tr \bP^{-1} \x - 2 \x\tr \bP^{-1} \hat{\x} + \text{const}_3
\, .
\end{align}$$
Eqns (5) and (6) follow by identification.
''']


# Also comment on CFL condition (when resolution is increased)?
# Excessive spacing needed for Colab to make list.
answers['Cov memory'] = ['MD',r'''


 - (a). $M$-by-$M$
 - (b). Using the [cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation),
    at least 2 times $M^3/3$.
 - (c). Assume $\B$ stored as float (double). Then it's 8 bytes/element.
        And the number of elements in $\B$: $M^2$. So the total memory is $8 M^2$.
 - (d). 8 trillion bytes. I.e. 8 million MB.
''']


answers['Woodbury'] = ['MD',r'''
We show that they cancel:
$$
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
The corollary follows from the Woodbury identity
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
A straightforward validation of (C2)
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

answers['KDE'] = ['MD',r'''
    from scipy.stats import gaussian_kde
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
 2. Compute the average error ("bias") of $\bx$. Verify that it converges to 0 as $N$ is increased.
 3. Compute the average *squared* error. Verify that it is approximately $\text{diag}(\B)/N$.
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


 * (a). Note that $\E \ones / N = \bx$.  
 And that $\bx \ones^T = \begin{bmatrix} \bx, & \ldots & \bx \end{bmatrix} \, .$  
 Use this to write out $\E \AN$.
 * (b). Show that element $(i,j)$ of the matrix product $\X^{} \Y^T$  
 equals element $(i,j)$ of the sum of the outer product of their columns:
 $\sum_n \x_n \y_n^T$.  
 Put this in the context of $\barB$.
 * (c). Use the following code:

...

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
answers["EnKF_nobias_a"] = ['MD',r'''
Let $\ones$ be the vector of ones of length $N$. Then
$$\begin{align}
    \bx^a
    &= \frac{1}{N} \E^\tn{a} \ones \tag{because $\sum_{n=1}^N \x^\tn{a}_n = \E^\tn{a} \ones$.} \\\
    &= \frac{1}{N} \E^\tn{f} \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Dobs - \bH \E^\tn{f} \right) \ones \tag{inserting eqn. (4).}
\end{align}$$
Assuming $\Dobs \ones=\\bvec{0}$ yields eqn. (6).
One might say that the mean of the EnKF update conforms to the KF mean update.  

"Conforming" is not a well-defined math word.
However, the above expression makes it clear that $\bx^\tn{a}$ is linear with respect to $\Dobs$, so that
$$\begin{align}
    \Expect \bx^\tn{a}
    &= \frac{1}{N} \E^\tn{f} \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Expect\Dobs - \bH \E^\tn{f} \right) \, .
\end{align}$$

Now, since $\Expect \br_n = \bvec{0}$, it follows that $\Expect \Dobs = \bvec{0}$,
and we recover eqn. (6).

The conclusion: the mean EnKF update is unbiased...
However, this is only when $\E^\tn{f}$ is considered fixed, and its moments assumed correct.
''']

answers["EnKF_nobias_b"] = ['MD',r'''
First, compute the updated anomalies, $\X^\tn{a}$, by inserting  eqn. (4) for $\E^a$:
$$\begin{align}
	\X^\tn{a}
	&= \E^a \AN \\\
	%&= {\X} + \barK\left(\y \ones\tr - \D - \bH \E^f\right) \AN \\\
	&= {\X} - \barK \left[\D + \bH \X\right] \, , \tag{A1}
\end{align}$$
where the definition of $\D$ has been used.

Inserting eqn. (A1) for the updated ensemble covariance matrix, eqn. (7b):
$$\begin{align}
	\barP
	&= \frac{1}{N-1} \X^\tn{a}{\X^\tn{a}}\tr \\\
    %
	&= \frac{1}{N-1} \left({\X} - \barK\left[\D + \bH \X\right]\right)
	\left({\X} - \barK\left[\D + \bH \X\right]\right)\tr \, .  \tag{A2}
\end{align}$$

Writing this out, and employing
the definition of $\barB$, eqn. (7a), yields:
$$\begin{align}
    \barP
	&= \barB + \barK \bH \barB \bH\tr \barK{}\tr -  \barB \bH\tr \barK{}\tr - \barK \bH \barB \tag{A3} \\\
	&\phantom{= } - \frac{1}{N-1}\left[
		{\X}\D\tr \barK{}\tr
		+ \barK\D {\X}\tr
		- \barK\D {\X}\tr \bH\tr \barK{}\tr
		- \barK \bH \X \D\tr \barK{}\tr
		- \barK\D\D\tr \barK{}\tr
	\right] \notag \, .
\end{align}$$

Substituting eqns. (9) into eqn. (A3) yields
$$\begin{align}
	\barP
	&=  \barB  + \barK \bH \barB \bH\tr \barK{}\tr -  \barB \bH\tr \barK{}\tr - \barK \bH \barB  + \barK \R \barK{}\tr \tag{A4} \\\
	&=  (\I_M - \barK \bH) \barB + \barK(\bH \barB \bH\tr + \R)\barK{}\tr -  \barB \bH\tr \barK{}\tr \tag{regrouped.} \\\
	&=  (\I_M - \barK \bH) \barB + \barB \bH\tr \barK{}\tr -  \barB \bH\tr \barK{}\tr \, , \tag{inserted eqn. (5a).} \\\
    &=  (\I_M - \barK \bH) \barB \, . \tag{10}
    \end{align}$$
Thus the covariance of the EnKF update "conforms" to the KF covariance update.

Finally, eqn. (A3) shows that
$\barP$ is linear with respect to the objects in eqns. (9).
Moreover, with $\Expect$ prepended, eqns. (9) hold true (not just as an assumption).
Thus, $\Expect \barP$ also equals eqn. (10).

The conclusion: the analysis/posterior/updated covariance produced by the EnKF is unbiased
(in the same, limited sense as for the previous exercise.)
''']

answers["EnKF_without_perturbations"] = ['MD',r'''
If $\Dobs = \bvec{0}$, then eqn. (A3) from the previous answer becomes
$$\begin{align}
    \barP
	&= (\I_M-\barK \bH)\barB(\I_M-\bH\tr \barK{}\tr) \tag{A5} \, ,
\end{align}$$
which shows that the updated covariance would be too small.
''']


answers['EnKF v1'] = ['MD',r'''
    def my_EnKF(N):
        E = mu0[:,None] + P0_chol @ randn((M,N))
        for k in range(1,K+1):
            # Forecast
            t   = k*dt
            E   = Dyn(E,t-dt,dt)
            E  += Q_chol @ randn((M,N))
            if k%dkObs == 0:
                # Analysis
                y        = yy[k//dkObs-1] # current obs
                Eo       = Obs(E,t)
                BH       = estimate_cross_cov(E,Eo)
                HBH      = estimate_mean_and_cov(Eo)[1]
                Perturb  = R_chol @ randn((p,N))
                KG       = divide_1st_by_2nd(BH, HBH+R)
                E       += KG @ (y[:,None] - Perturb - Eo)
            xxhat[k] = mean(E,axis=1)
''']

answers['rmse'] = ['MD',r'''
    rmses = sqrt(np.mean((xx-xxhat)**2, axis=1))
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
$ \newcommand{\x}{\x} \|\x\|_2 \leq M^{1/2} \|\x\|\_\infty \text{and}  \|\x\|_1 \leq M^{1/2} \|\x\|_2$
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
