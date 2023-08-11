# You can edit answers here and preview in notebook by using
#
# >>> from resources import answers
# >>> import importlib
# >>> importlib.reload(answers)
#
# Alternatively you can edit them in the notebook as follows.
#
# >>> import resources.answers as aaa
# >>> aaa.answers['name'] = ['MD', r'''
# >>> ...
# >>> ''']
# >>> ws.show_answer('name')

from markdown import markdown as md2html # better than markdown2 ?
from IPython.display import HTML, display

from .macros import include_macros

def show_answer(tag, *subtags):
    """Display answer corresponding to 'tag' and any 'subtags'."""

    # Homogenize, e.g. 'a, b cd' --> 'abcd'
    subtags = ", ".join(subtags)
    subtags.translate(str.maketrans('', '', ' ,'))

    for key in filter(lambda key: key.startswith(tag), answers):
        if not subtags or any(key.endswith(" "+ch) for ch in subtags):
            formatted_display(*answers[key], '#dbf9ec') #d8e7ff


def formatted_display(TYPE, content, bg_color):
    # ALWAYS remove 1st linebreak -- my convention
    content = content[1:]

    # Convert from TYPE to HTML
    if TYPE == "TXT":
        content = '<pre><code>'+content+'</code></pre>'
    elif TYPE == "MD":
        # For some reason, rendering list-only content requires newline padding
        if any(content.lstrip().startswith(bullet) for bullet in "-*"):
            content = "\n\n" + content
        content = md2html(include_macros(content))
    else:
        pass # assume already html

    # Make bg style
    bg_color = '#dbf9ec'  # 'd8e7ff'
    bg_color = 'background-color:'+ bg_color + ';' #d8e7ff #e2edff

    # Compose string
    content = '<div style="'+bg_color+'padding:0.5em;">'+str(content)+'</div>'

    # Fix Colab - MathJax incompatibility
    setup_typeset()

    # Display
    display(HTML(content))


# TODO: obsolete?
def setup_typeset():
    """MathJax initialization for the current cell.

    This installs and configures MathJax for the current output.

    Necessary in Google Colab. Ref:
    https://github.com/googlecolab/colabtools/issues/322
    """

    # Only run in Colab
    try:
        import google.colab  # type: ignore
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
answers['thesaurus 1'] = ["TXT", r"""
Ensemble      Stochastic     Data
Sample        Random         Measurements
Set of draws  Monte-Carlo    Observations
"""]

answers['thesaurus 2'] = ["TXT", r"""
Statistical inference    Ensemble member     Quantitative belief    Recursive
Inverse problems         Sample point        Probability            Sequential
Inversion                Realization         Relative frequency     Iterative
Estimation               Single draw         Estimate               Serial
Regression               Particle            Information
Fitting                                      Uncertainty
                                             Knowledge
"""]

answers['Discussion topics 1'] = ['MD', r'''
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
# Tut: Bayesian inference & Gaussians
###########################################
answers['pdf_G1'] = ['MD', r'''
    const = 1/np.sqrt(2*np.pi*sigma2)
    pdf_values = const * np.exp(-0.5*(x - mu)**2/sigma2)
''']

answers['Gauss integrals'] = ['MD', r'''
(i) $$\begin{align} \Expect[x]
&= \int x \, c \, e^{-(x-\mu)^2 / 2 \sigma^2} \,d x \tag{by definition} \\\
&= \int (u + \mu) \, c \, e^{-u^2 / 2 \sigma^2} \,d u \tag{$u = x-\mu$}\\\
&= \int u \, c \, e^{-u^2 / 2 \sigma^2}
+       \mu \, c \, e^{-u^2 / 2 \sigma^2} \,d u \\\
&= [-\sigma^2 \, c \, e^{-u^2 / 2 \sigma^2}]^{+\infty}_{-\infty}
+ \mu \, \Expect[1]
\end{align}
$$
The first term is zero. The second leaves only $\mu$, since $\Expect[1] = 1$.

(ii) $$\begin{align} \Expect[(x - \mu)^2]
&= \int (x - \mu)^2 \, c \, e^{-(x-\mu)^2 / 2 \sigma^2} \,d x \tag{by definition} \\\
&= \int u^2 \, c \, e^{-u^2 / 2 \sigma^2} \,d u \tag{$u = x-\mu$}\\\
&= \int u \, u \, c \, e^{-u^2 / 2 \sigma^2} \,d u \\\
&= 0 - \int (1) (-\sigma^2) \, c \, e^{-u^2 / 2 \sigma^2} \,d u \,,  \tag{Integrate by parts} \\\
\end{align}
$$
where the first term was zero for the same raeson as above,
and the second can again be expressed in terms of $\Expect[1] = 1$.
''']

answers['Why Gaussian'] =  ['MD', r"""
What's not to love? Consider

 * The central limit theorem (CLT) and all its implications.
 * Pragmatism: Yields "least-squares problems", whose optima is given by a linear systems of equations.  
 * Self-conjugate: Gaussian prior and likelihood yields Gaussian posterior.
 * Among pdfs with independent components (2 or more),
   the Gaussian is uniquely (up to scaling) rotation-invariant (symmetric).
 * Uniquely for Gaussian sampling distribution: maximizing the likelihood for the mean simply yields the sample average.
 * For more, see [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution#Properties)
   and Chapter 7 of: [Probability theory: the logic of science (Edwin T. Jaynes)](https://books.google.com/books/about/Probability_Theory.html?id=tTN4HuUNXjgC).
"""]


answers['GG BR example'] = ['MD', r'''
- Eqn. (5) yields $P\supa = \frac{1}{1/4 + 1/4} = \frac{1}{2/4} = 2$.
- Eqn. (6) yields $x\supa = 2 \cdot (20/4 + 18/4) = \frac{20 + 18}{2} = 19$
''']

answers['symmetry of conditioning'] = ['MD', r'''
<a href="https://en.wikipedia.org/wiki/Bayes%27_theorem#For_continuous_random_variables" target="_blank">Wikipedia</a>

''']

answers["what's forward?"] = ['MD', r'''
Because estimation/inference/inverse problems
are looking for the "cause" from the "outcomes".
Of course, causality is a tricky notion.
Anyway, these problems tend to be harder,
generally because they involve solving (possibly approximately) a set of equations
rather than just computing a formula (corresponding to a forward problem).
Since $y = f(x)$ is common symbolisism for formulae,
it makes sense to use the symobls $x = f^{-1}(y)$ for the estimation problem.
Note that, despite the "glorious" ideas invoked by language such as "inverse/inversion",
all techniques (known to author) still essentially come down to
some form of *fitting* to data/observations.
''']

answers['Posterior behaviour'] = ['MD', r'''
- Likelihood becomes flat.  
  Thus, the posterior is dominated by the prior,
  and becomes (in the limit) superimposed on it.
- Likelihood becomes a delta function.  
  Posterior gets dominated by the likelihood, and superimposed on it.
- It's located halfway between the prior/likelihood $\forall y$.
- It's always higher than the height of the prior and the likelihood.
  This even holds true for some interval in the mixed Gaussian-uniform cases.
- "Information-wise", the "expected" posterior entropy (and variance)
  is always smaller than that of the prior,
  but in practice (i.e. without averaging over the observations)
  we can observe that it is not necessarily the case.
    - For the uniform-uniform case, yes.
    - For the mixed case, it's not clear what scale/width means,
      but for most (any reasonable?) definitions the answer will be yes.
    - For the Gaussian-Gaussian case, no.
- In the mixed cases: Yes. Otherwise: No.
- In fact, we'll see later that the Gaussian-Gaussian posterior is Gaussian,
  and is therefore fully characterized by its mean and its variance.
  So we only need to compute its location and scale.
- The problem (of computing the posterior) is ill-posed:  
  The prior says there's zero probability of the truth being 
  in the region where the likelihood is not zero.
- This of course depends on the definition of "sufficient",
  but in the authors opinion $N \geq 40$ seems necessary.
''']

answers['quadrature marginalisation'] = ['MD', r'''
$$\texttt{sum(pp)*dx}
\approx \int \texttt{pp}(x) \, dx
= \int p(x) \, p(y|x) \, dx
= \int p(x,y) \, dx
= p(y) \, .$$
''']

answers['nD-space is big a'] = ['MD', r'''
$N^{\\xDim}$
''']
answers['nD-space is big b'] = ['MD', r'''
$15 * 360 * 180 = 972'000 \approx 10^6$
''']
answers['nD-space is big c'] = ['MD', r'''
$20^{10^6}$

For comparison, there are about $10^82$ atoms in the universe.
''']

answers['BR Gauss, a.k.a. completing the square a'] = ['MD', r'''
Expanding the squares of the left hand side (LHS),
and gathering terms in powers of $x$ yields
$$
    \frac{(x-x\supf)^2}{P\supf} + \frac{(x-y)^2}{R}
    =  x^2 (1/P\supf + 1/R)
    - 2 x (x\supf/P\supf + y/R)
    + c_1
    \,, \tag{a1}
$$
with $c_1 = (x\supf)^2/P\supf + y^2/R$.
Meanwhile
$$
    \frac{(x-x\supa)^2}{P\supf}
    = x^2 / P\supa
    - 2 x x\supa/P\supa
    + (x\supa)^2/P\supa
    \,.
\tag{a2}
$$
Both (a1) and (a2) are quadratics in $x$,
so we can equate them by setting
$$ \begin{align}
1/P\supa = 1/P\supf + 1/R \,, \tag{a3} \\\
x\supa/P\supa = x\supf/P\supf + y/R \,, \tag{a4}
\end{align} $$
whereupon we immediately recover $P\supa$ and $x\supa$ of eqns. (5) and (6).

*PS: The above process is called "completing the square"
since it involves writing a quadratic polynomial as a single squared term
plus a "constant" that we add and subtract.*
''']

answers['BR Gauss, a.k.a. completing the square b'] = ['MD', r'''
From part (a),
$$
    \frac{(x-x\supf)^2}{P\supf} + \frac{(x-y)^2}{R}
    =
    \frac{(x-x\supa)^2}{P\supf} + c_2
    \,,
\tag{a5}
$$
with $c_2 = c_1 - (x\supa)^2/P\supa$.
Substituting in the formulae for $c_1$ and $x\supa$ produces
$$
c_2 = (x\supf)^2/P\supf + y^2/R - P\supa (x\supf/P\supf + y/R)^2
= y^2 ( 1/R - P\supa/R^2 ) - 2 y x\supf \frac{P\supa}{P\supf R} + \frac{(x\supf)^2}{P\supf} - P\supa \frac{(x\supf)^2}{(P\supf)^2}  
\tag{a6}
$$
Now, multiplying eqn. (a3) with $P\supf R$, it can be seen that

- $\frac{P\supa}{P\supf R} = \frac{1}{P\supf + R}$, whence
- $\frac{P\supa}{P\supf} = \frac{R}{P\supf + R}$, and
- $\frac{P\supa}{R} = \frac{P\supf}{P\supf + R}$ so that
- $1/R - P\supa/R^2 = \frac{1}{R}(1 - \frac{P\supa}{R} ) = \frac{1}{R} \frac{R}{P\supf + R} = \frac{1}{P\supf + R}$.

Thus eqn. (a6) simplifies to
$$ \begin{align}
c_2
&= y^2 \frac{1}{P\supf + R}  - 2 y x\supf \frac{1}{P\supf + R} + \frac{(x\supf)^2}{P\supf} - \frac{(x\supf)^2}{P\supf}\frac{R}{P\supf + R} \\\
&= \frac{1}{P\supf + R} \Bigl[ y^2 - 2 y x\supf + (x\supf)^2 \bigl( \frac{P\supf + R}{P\supf} - \frac{R}{P\supf}\bigr ) \Bigr ] \\\
&= \frac{(y - x\supf)^2}{P\supf + R}
\tag{a7}
\end{align} $$
''']

answers['BR Gauss, a.k.a. completing the square c'] = ['MD', r'''
\begin{align}
p(x|y)
&\propto p(x) \, p(y|x) \\\
&=       N(x \mid x\supf, P\supf) \, N(y \mid x, R) \\\
&\propto \exp \Big( \frac{-1}{2} \big[ (x-x\supf)^2/P\supf + (x-y)^2/R \big] \Big) \,.
\end{align}
The rest follows by eqn. (S2) and identification with $N(x \mid x\supa, P\supa)$.
''']

answers['BR Gauss'] = ['MD', r'''
We can ignore factors that do not depend on $x$.

\begin{align}
p(x|y)
&= \frac{p(x) \, p(y|x)}{p(y)} \\\
&\propto p(x) \, p(y|x) \\\
&=       N(x \mid x\supf, P\supf) \, N(y \mid x, R) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (x-x\supf)^2/P\supf + (x-y)^2/R \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (1/P\supf + 1/R)x^2 - 2(x\supf/P\supf + y/R)x \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( x - \frac{x\supf/P\supf + y/R}{1/P\supf + 1/R} \Big)^2 \cdot (1/P\supf + 1/R) \Big) \, .
\end{align}

Identifying the last line with $N(x \mid x\supa, P\supa)$ yields eqns (5) and (6).
''']

answers['BR Kalman1 algebra'] = ['MD', r'''
- Multiplying eqn. (5) by $1 = \frac{P\supf R}{P\supf R}$ yields
  $P\supa = \frac{P\supf R}{P\supf + R} = \frac{P\supf}{P\supf + R} R$, i.e. eqn. (8).
- Alternatively, $P\supa = \frac{R}{P\supf + R} P\supf$.
  Adding $0 = P\supf - P\supf$ to the numerator yields eqn. (10).
- Applying formulae (8) and (10) for $P\supa$
  in eqn. (6) immediately produces eqn. (11).
''']

answers['KG intuition'] = ['MD', r'''
Consider eqn. (9). Both nominator and denominator are strictly larger than $0$,
hence $K > 0$. Meanwhile $P\supf + R > P\supf$, hence $K<1$.

Since $0<K<1$, eqn. (8) yields $P\supa < R$,
while eqn. (10) yields $P\supa < P\supf$.

From eqn. (11), $x\supa = (1-K) x\supf + K y$.
Since $0<K<1$, we can see that $x\supa$
is a 'convex combination' or 'weighted average'.
*For even more detail, consider the case $x\supf<y$ and then case $y<x\supf$.*

Because it describes how much the esimate is "dragged" from $x\supf$ "towards" $y$.  
I.e. it is a multiplification (amplification) factor,
which French (signal processing) people like to call "gain".  

Relatedly, note that $K$ weights the observation uncertainty $(R)$ vs. the total uncertainty $(P\supf + R)$,
and so is always between 0 and 1.
''']

answers['BR Kalman1 code'] = ['MD', r'''
    KG = Pf / (Pf + R)
    Pa = (1 - KG) * Pf
    xa = xf + KG * (y - xf)
''']

answers['Posterior cov'] =  ['MD', r"""
  * No.
      * It means that information is always gained.
      * No, not always.  
        But on average, yes:  
        [the "expected" posterior entropy (and variance)
        is always smaller than that of the prior.](https://www.quora.com/What-conditions-guarantee-that-the-posterior-variance-will-be-less-than-the-prior-variance#)
  * It probably won't have decreased. Maybe you will just discard the new information entirely, in which case your certainty will remain the same.  
    I.e. humans are capable of thinking hierarchically, which effectively leads to other distributions than the Gaussian one.
"""]

###########################################
# Tut: Univariate Kalman filtering
###########################################

answers['AR1'] = ['MD', r'''
- The seed controls the random numbers generated via `numpy.random`
  (imported as `rnd`) that are used to generate the stochastic processes,
  i.e. the `truth` $\\{x_k\\}$ and `obsrvs` $\\{y_k\\}$.
-   - `0`: $\\{x_k\\}$ becomes [(Gaussian) white noise](https://en.wikipedia.org/wiki/White_noise#Discrete-time_white_noise).  
      But if $Q = 0$, it just becomes the constant $0$.
    - `1`: $\\{x_k\\}$ becomes a [(Gaussian) random walk](https://en.wikipedia.org/wiki/Random_walk#Gaussian_random_walk),
      a.k.a. Brownian motion a.k.a. Wiener process.  
      But if $Q=0$ it just becomes the constant $x_0$.
    - As `M` approaches 1 (from below), the process becomes more and more auto-correlated.  
      Its variance (at any given time index) also increases.  
      But if $Q=0$ then the process just decays (geometrically/exponentially) to $0$.
    - `>1`: The process becomes a divergent geometric sequence, and blows up (for any value of $Q$).
- The observations become exact (infinitely precise and accurate).
- The observations become useless, carrying no information.  
  Their variance dominates in the plot, so that it **looks** like $\\{x_k\\}$ becomes flat.
''']

answers['KF behaviour'] = ['MD', r'''
- The uncertainty never reaches 0 (but decays geometrically in time).
- The dynamics always produces 0 at the next time step
  (also since there is almost no noise).
  Therefore, regardless of initial variance or observation precision,
  the KF knows exactly where the state is: at 0.
  Happily, this is reflected in its uncertainty estimate,
  i.e. variance, which is also 0.
''']

answers['KF with bias'] = ['MD', r'''
-   - If `logR_bias` $\rightarrow -\infty$ then the KF "trusts" the observations too much,
    always jumping to lie right on top of them (also assuming `H=1`).
    - The same happens for `logQ_bias` $\rightarrow +\infty$
    since then the KF has no "faith" in its predictions
    (easier to see if you comment out the line that plots the uncertainty which is now humongous).
    - If `logR_bias` $\rightarrow +\infty$ then the KF attributes no weight to the observations,
    so that only the mean model prediction matters, which is zero.
    - The same happens for `logQ_bias` $\rightarrow -\infty$, except for an initial transitory
    period during which the initial uncertainty of the KF quickly attenuates,
    after which it only trusts its mean prediction.
    - *PS: it may appear that $R$ and $Q$ simply play the opposite role
    (while adding to the complexity of the KF). However, this is only approximately true,
    and less so in more complex settings.*
- Not very long? But longer for larger $R$ or smaller $Q$.
- Even longer.
''']

# Also see 'Gaussian sampling a'
answers['RV sums'] = ['MD', r'''
By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
and that of (Dyn),
the mean parameter becomes:
$$ \\Expect(\\DynMod x+q) =  \\DynMod \\Expect(x) + \\Expect(q) = \\DynMod x\supa \, . $$

Moreover, by independence,
$ \\text{Var}(\\DynMod x+q) = \\text{Var}(\\DynMod x) + \\text{Var}(q) $,
and so
the variance parameter becomes:
$$ \\text{Var}(\\DynMod x+q) = \\DynMod^2 P\supa + Q \, .  $$
''']

answers['LinReg deriv a'] = ['MD', r'''
$$ \begin{align}
p\, (y_1, \ldots, y_K \;|\; a)
&= \prod_{k=1}^K \, p\, (y_k \;|\; a) \tag{each obs. is indep. of others, knowing $a$.} \\\
&= \prod_k \, \mathcal{N}(y_k \mid a k,R) \tag{inserted eqn. (3) and then (1).} \\\
&= \prod_k \,  (2 \pi R)^{-1/2} e^{-(y_k - a k)^2/2 R} \tag{inserted eqn. (G1) from T2.} \\\
&= c \exp\Big(\frac{-1}{2 R}\sum_k (y_k - a k)^2\Big) \\\
\end{align} $$
Taking the logarithm is a monotonic transformation, so it does not change the location of the maximum location.
Neither does diving by $c$. Multiplying by $-2 R$ means that the maximum becomes the minimum.
''']

answers['LinReg deriv b'] = ['MD', r'''
$$ \frac{d J_K}{d \hat{a}} = 0 = \ldots $$
''']

answers['LinReg_k'] = ['MD', r'''
    kk = 1+np.arange(k)
    a = sum(kk*yy[kk]) / sum(kk**2)
''']

answers['Sequential 2 Recursive'] = ['MD', r'''
    (k+1)/k
''']

answers['LinReg ⊂ KF'] = ['MD', r'''
Linear regression is only optimal if the truth is a straight line,
i.e. if $\\DynMod_k = (k+1)/k$.  

Compared to the KF, which accepts a general $\\DynMod_k$,
this is so restrictive that one does not usually think
of the methods as belonging to the same class at all.
''']

answers['LinReg compare'] = ['MD', r'''
Let $\hat{a}_K$ denote the linear regression estimates of the slope $a$
based on the observations $y_1, \ldots, y_K$.  
Let $x\supa_K$ denote the KF estimate of $x\supa_K$ based on the same set of obs.  
It can bee seen in the plot that
$ x\supa_K = K \hat{a}_K \, . $
''']

answers['x_KF == x_LinReg'] = ['MD', r'''
We'll proceed by induction.  

With $P\supf_1 = \infty$, we get $P\supa_1 = R$,
which initializes (13).  

Now, inserting (13) in (12) yields:

$$
\begin{align}
P\supa_{K+1} &= 1\Big/\big(1/R + \textstyle (\frac{K}{K+1})^2 / P\supa_K\big)
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

The proof for $x\supa_k$ is similar.
''']


answers['KF1 code'] = ['MD', r'''

            # Forecast step
            xf = M * xa
            Pf = M**2 * Pa + Q
            # Analysis update step
            Pa = 1 / (1/Pf + H**2/R)
            xa = Pa * (xf/Pf + H*obsrvs[k]/R)

''']

answers['Asymptotic Riccati a'] = ['MD', r'''
Merging forecast and analysis equations for $P\supa_k$,
and focusing on their inverses (called "precisions")
we find
$$ 1/P\supa_k = 1/(M^2 P\supa_{k-1}) + H^2/R \,,$$

Note that $P\supa_k < P\supf_k$ for each $k$
(c.f. [Gaussian-Gaussian Bayes rule](T3%20-%20Bayesian%20inference.ipynb#Exc----GG-Bayes))
Thus,
$$
P\supa_k < P\supf_k = \\DynMod^2 P\supa_{k-1}
\xrightarrow[k \rightarrow \infty]{} 0 \, .
$$
''']

answers['Asymptotic Riccati b'] = ['MD', r'''
Since
$ 1/P\supa_k = 1/P\supa_{k-1} + 1/R \,,$
it follows that
$ 1/P\supa_k = 1/P\supa_0 + k / R \xrightarrow[k \rightarrow \infty]{} +\infty \,,\quad$
i.e.
$ P\supa_k \rightarrow 0 \,.$
''']

answers['Asymptotic Riccati c'] = ['MD', r'''
The fixed point $P\supa_\infty$ should satisfy
$P\supa_\infty = 1/\big(1/R + 1/[\\DynMod^2 P\supa_\infty]\big)$,
yielding $P\supa_\infty = R (1-1/\\DynMod^2)$.  
Note that this is **not 0**!
In other words, even though the KF keeps gaining observational data/information,
this gets balanced out by the growth in error/uncertainty during the forecast.
Also note that the asymptotic state uncertainty ($P\supa_\infty$)
is directly proportional to the observation uncertainty ($R$).
''']

answers['signal processing a'] = ['MD', r'''
    sigproc['Wiener']   = sig.wiener(obsrvs)
''']
answers['signal processing b'] = ['MD', r'''
    sigproc['Hamming']  = sig.convolve(obsrvs, nrmlz(sig.windows.hamming(10)), 'same')
''']
answers['signal processing c'] = ['MD', r'''
    sigproc['Low-pass'] = np.fft.irfft(trunc(np.fft.rfft(obsrvs), len(obsrvs)//4))
''']
answers['signal processing d'] = ['MD', r'''
    sigproc['Butter']   = sig.filtfilt(*sig.butter(4, .5), obsrvs, padlen=10)
''']
answers['signal processing e'] = ['MD', r'''
    sigproc['Spline']   = sp.interpolate.UnivariateSpline(kk, obsrvs, s=1e2)(kk)
''']

###########################################
# Tut: Time series analysis
###########################################


###########################################
# Tut: Multivariate Kalman
###########################################

answers['Likelihood for additive noise'] = ['MD', r'''
Start by assuming (in place of eqn 2) that $\y=\br$;
then $\y$ would have the same distribution as $\br$.
Adding $\ObsMod(\x)$ just shifts (translates) the distribution.
More formally, one can use the fact that the pdf $p(\y)$
can be defined by the CDF as $\frac{\partial}{\partial \y} \mathbb{P}(\\{Y_i \leq y_i\\})$,
and insert eqn. (2) for $\mathbf{Y}$.
In summary, the addition of $\ObsMod(\x)$ merely changes the mean,
hence $\mathcal{N}(\y \mid \ObsMod(\x), \R)$.

One can also derive the likelihood using only pdfs.
The problem with this derivation is that one first needs to justify
the use of delta functions to descrie deterministic relationships.
$$
\begin{align}
p(\y|\x)
&= \int p(\y, \br|\x) \, d \br \tag{by law of total proba.}  \\\
&= \int p(\y|\br, \x) \, p(\br|\x) \, d \br \tag{by def. of conditional proba.} \\\
&= \int \delta\big(\y-(\ObsMod(\x) + \br)\big) \, p(\br|\x) \, d \br \tag{$\y$ is fully determined by $\x$ and $\br$} \\\
&= \int \delta\big(\y-(\ObsMod(\x) + \br)\big) \, \mathcal{N}(\br \mid \\bvec{0}, \R) \, d \br \tag{the draw of $\br$ does not depened on $\x$} \\\
&= \mathcal{N}(\y - \ObsMod(\x) \mid \\bvec{0}, \R) \tag{by def. of Dirac Delta} \\\
&= \mathcal{N}(\y \mid \ObsMod(\x), \R) \tag{by reformulation} \, .
\end{align}
$$
''']

answers['KF precision'] = ['MD', r'''
By Bayes' rule:
$$\begin{align}
- 2 \log p(\x|\y) =
\|\bH \x-\y \|\_\R^2 + \| \x - \bb \|\_{\bP\supf}^2
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
answers['nD-covars are big'] = ['MD', r'''
 - (a). ${\\xDim}$-by-${\\xDim}$
 - (b). Using the [cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation),
    at least 2 times ${\\xDim}^3/3$.
 - (c). Assume ${\bP\supf}$ stored as float (double). Then it's 8 bytes/element.
        And the number of elements in ${\bP\supf}$: ${\\xDim}^2$. So the total memory is $8 {\\xDim}^2$.
 - (d). 8 trillion bytes. I.e. 8 million MB.
''']


answers['Woodbury'] = ['MD', r'''
We show that they cancel:
$$
\begin{align}
  &\left(\B^{-1}+\V\tr \R^{-1} \U \right)
  \left[ \B - \B \V\tr \left(\R+\U \B \V\tr \right)^{-1} \U \B \right] \\\
  & \quad = \I_{\\xDim} + \V\tr \R^{-1}\U \B -
  (\V\tr + \V\tr \R^{-1} \U \B \V\tr)(\R + \U \B \V\tr)^{-1}\U \B \\\
  & \quad = \I_{\\xDim} + \V\tr \R^{-1}\U \B -
  \V\tr \R^{-1}(\R+ \U \B \V\tr)(\R + \U \B \V\tr)^{-1} \U \B \\\
  & \quad = \I_{\\xDim} + \V\tr \R^{-1} \U \B - \V\tr \R^{-1} \U \B \\\
  & \quad = \I_{\\xDim}
\end{align}
$$
''']

answers['inv(SPD + SPD)'] = ['MD', r'''
The corollary follows from the Woodbury identity
by replacing $\V, \U$ by $\bH$,
*provided that everything is still well-defined*.
In other words,
we need to show the existence of the left hand side.

Now, for all $\x \in \Reals^{\\xDim}$, $\x\tr \B^{-1} \x > 0$ (since $\B$ is SPD).
Similarly, $\x\tr \bH\tr \R^{-1} \bH \x\geq 0$,
implying that the left hand side is SPD:
$\x\tr (\bH\tr \R^{-1} \bH + \B^{-1})\x > 0$,
and hence invertible.
''']

answers['Woodbury C2'] = ['MD', r'''
A straightforward validation of (C2)
is obtained by cancelling out one side with the other.
A more satisfying exercise is to derive it from (C1)
starting by right-multiplying by $\bH\tr$.
''']


###########################################
# Tut: Dynamical systems, chaos, Lorenz
###########################################

answers["Ergodicity a"] = ["MD", r'''
For asymptotically large $T$, the answer is "yes";
however, this is difficult to distinguish if $T<60$ or $N<400$,
which takes a very long time with the integrator used in the above.
''']

answers["Ergodicity b"] = ["MD", r'''
It doesn't matter
(provided the initial conditions for each experiment is "cropped out" before averaging).
''']

answers["Hint: Lorenz energy"] = ["MD", r'''
Hint: what's its time-derivative?
''']

answers["Lorenz energy"] = ["MD", r'''
\begin{align}
\frac{d}{dt}
\sum_i
x_i^2
&=
2 \sum_i
x_i \dot{x}_i
\end{align}

Next, insert the quadratic terms from the ODE,
$
\dot x_i = (x_{i+1} − x_{i-2}) x_{i-1}
\, .
$

Finally, apply the periodicity of the indices.
''']

answers["error evolution"] = ["MD", r"""
$\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
"""]
answers["anti-deriv"] = ["MD", r"""
Differentiate $e^{F t}$.
"""]
answers["predictability cases"] = ["MD", r"""
* (1). Dissipates to 0.
* (2). No.
      A balance is always reached between
      the uncertainty reduction $(1-K)$ and growth $F^2$.  
"""]
answers["saturation term"] = ["MD", r"""
[link](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
"""]
answers["liner growth"] = ["MD", r"""
$\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
"""]

answers["doubling time"] = ["MD", r"""
    xx   = output_63[0][:, -1]     # Ensemble of particles at the end of integration
    v    = np.var(xx, axis=0)      # Variance (spread^2) of final ensemble
    v    = np.mean(v)              # homogenize
    d    = np.sqrt(v)              # std. dev.
    eps  = [FILL IN SLIDER VALUE]  # initial spread
    T    = [FILL IN SLIDER VALUE]  # integration time
    rate = np.log(d/eps)/T         # assuming d = eps*exp(rate*T)
    print("Doubling time (approx):", log(2)/rate)
"""]


###########################################
# Tut: Geostats
###########################################

answers["nearest neighbour interp"] = ["MD", r"""
    nearest_obs = [np.argmin(d) for d in dists_xy]
    # nearest_obs = np.argmin(dists_xy, 1)
"""]

answers["inv-dist weight interp"] = ["MD", r"""
    weights = 1/dists_xy**exponent
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize
"""]

answers['Kriging code'] = ["MD", r"""
    covar_yy = 1 - variogram(dists_yy, **vg_params)
    cross_xy = 1 - variogram(dists_xy, **vg_params)
    regression_coefficients = sla.solve(covar_yy, cross_xy.T).T
"""]



###########################################
# Tut: Ensemble representation
###########################################

answers['KDE'] = ['MD', r'''
    from scipy.stats import gaussian_kde
    ax.plot(xx, gaussian_kde(E.ravel()).evaluate(xx), label="KDE estimate")
''']

answers['Gaussian sampling a'] = ['MD', r'''
Type `rnd.randn??` in a code cell and execute it.
''']
answers['Gaussian sampling b'] = ['MD', r'''
    z = rnd.randn(xDim, 1)
    x = b + L @ z
''']

answers['Gaussian sampling c'] = ['MD', r'''
    E = b[:, None] + L @ rnd.randn(xDim, N)
    # Alternatives:
    # E = np.random.multivariate_normal(b, B, N).T
    # E = ( b + rnd.randn(N, xDim) @ L.T ).T
''']

answers['Average sampling error'] = ['MD', r'''
Procedure:

 1. Repeat the experiment many times.
 2. Compute the average error ("bias") of $\bx$. Verify that it converges to 0 as $N$ is increased.
 3. Compute the average *squared* error. Verify that it is approximately $\text{diag}(\B)/N$.
''']

answers['ensemble moments'] = ['MD', r'''
    x_bar = np.sum(E, axis=1)/N
    B_bar = np.zeros((xDim, xDim))
    for n in range(N):
        xc = (E[:, n] - x_bar)[:, None] # x_centered
        B_bar += xc @ xc.T
        #B_bar += np.outer(xc, xc)
    B_bar /= (N-1)
''']

answers['Why (N-1)'] = ['MD', r'''
Because it is the [unbiased](https://en.wikipedia.org/wiki/Variance#Sample_variance)
estimates in the case of an unknown mean.

*PS: in practice, in DA,
this is more of a convention than a requirement,
since its impact is overshadowed by that of repeat cycling,
as well as inflation and localisation.*
''']

answers['variance estimate statistics'] = ['MD', r'''
 * Visibly, the expected value (mean) of $1/\barB$ is not $1$,
   so $1/\barB$ is not unbiased. This is to be expected,
   since taking the reciprocal is a *nonlinear* operation.
 * The mean of $\barB$ is $1$ for any ensemble size.
 * The mean  of $1/\barB$ is infinite for $ N=2 $,
   and decreases monotonically as $ N $ increases,
   tending to $1$ as $ N $ tends to $+\infty$.

''']


answers['ensemble moments vectorized'] = ['MD', r'''
 * (a). Note that $\E \ones / N = \bx$.  
 And that $\bx \ones^T = \begin{bmatrix} \bx, & \ldots & \bx \end{bmatrix} \, .$  
 Use this to write out $\E \AN$.
 * (b). Show that element $(i, j)$ of the matrix product $\X^{} \Y^T$  
 equals element $(i, j)$ of the sum of the outer product of their columns:
 $\sum_n \x_n \y_n^T$.  
 Put this in the context of $\barB$.
 * (c). Use the following code:

...

    x_bar = np.sum(E, axis=1, keepdims=True)/N
    X     = E - x_bar
    B_bar = X @ X.T / (N-1)   
''']

# Skipped
answers['Why matrix notation'] = ['MD', r'''
   - Removes indices
   - Highlights the linear nature of many computations.
   - Tells us immediately if we're working in state space or ensemble space
     (i.e. if we're manipulating individual dimensions, or ensemble members).
   - Helps with understanding subspace rank issues
   - Highlights how we work with the entire ensemble, and not individual members.
   - Suggest a deterministic parameterization of the distributions.
''']

answers['estimate cross'] = ['MD', r'''
    def estimate_cross_cov(Ex, Ey):
        N = Ex.shape[1]
        assert N==Ey.shape[1]
        X = Ex - np.mean(Ex, axis=1, keepdims=True)
        Y = Ey - np.mean(Ey, axis=1, keepdims=True)
        CC = X @ Y.T / (N-1)
        return CC
''']

answers['errors'] = ['MD', r'''
 * (a). Error: discrepancy from estimator to the parameter targeted.
Residual: discrepancy from explained to observed data.
 * (b). Bias = *average* (i.e. systematic) error.
 * (c). [Wiki](https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship)
''']


###########################################
# Tut: Writing your own EnKF
###########################################
answers["EnKF_nobias_a"] = ['MD', r'''
Let $\ones$ be the vector of ones of length $N$. Then
$$\begin{align}
    \bx\supa
    &= \frac{1}{N} \E\supa \ones \tag{because $\sum_{n=1}^N \x\supa_n = \E\supa \ones$.} \\\
    &= \frac{1}{N} \E\supf \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Dobs - \bH \E\supf \right) \ones \tag{inserting eqn. (4).}
\end{align}$$
Assuming $\Dobs \ones=\\bvec{0}$ yields eqn. (6).
One might say that the mean of the EnKF update conforms to the KF mean update.  

"Conforming" is not a well-defined math word.
However, the above expression makes it clear that $\bx\supa$ is linear with respect to $\Dobs$, so that
$$\begin{align}
    \Expect \bx\supa
    &= \frac{1}{N} \E\supf \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Expect\Dobs - \bH \E\supf \right) \, .
\end{align}$$

Now, since $\Expect \br_n = \bvec{0}$, it follows that $\Expect \Dobs = \bvec{0}$,
and we recover eqn. (6).

The conclusion: the mean EnKF update is unbiased...
However, this is only when $\E\supf$ is considered fixed, and its moments assumed correct.
''']

answers["EnKF_nobias_b"] = ['MD', r'''
First, compute the updated anomalies, $\X\supa$, by inserting  eqn. (4) for $\E\supa$:
$$\begin{align}
	\X\supa
	&= \E\supa \AN \\\
	%&= {\X} + \barK\left(\y \ones\tr - \D - \bH \E\supf\right) \AN \\\
	&= {\X} - \barK \left[\D + \bH \X\right] \, , \tag{A1}
\end{align}$$
where the definition of $\D$ has been used.

Inserting eqn. (A1) for the updated ensemble covariance matrix, eqn. (7b):
$$\begin{align}
	\barP
	&= \frac{1}{N-1} \X\supa{\X\supa}\tr \\\
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
	&=  (\I_{\\xDim} - \barK \bH) \barB + \barK(\bH \barB \bH\tr + \R)\barK{}\tr -  \barB \bH\tr \barK{}\tr \tag{regrouped.} \\\
	&=  (\I_{\\xDim} - \barK \bH) \barB + \barB \bH\tr \barK{}\tr -  \barB \bH\tr \barK{}\tr \, , \tag{inserted eqn. (5a).} \\\
    &=  (\I_{\\xDim} - \barK \bH) \barB \, . \tag{10}
    \end{align}$$
Thus the covariance of the EnKF update "conforms" to the KF covariance update.

Finally, eqn. (A3) shows that
$\barP$ is linear with respect to the objects in eqns. (9).
Moreover, with $\Expect$ prepended, eqns. (9) hold true (not just as an assumption).
Thus, $\Expect \barP$ also equals eqn. (10).

The conclusion: the analysis/posterior/updated covariance produced by the EnKF is unbiased
(in the same, limited sense as for the previous exercise.)
''']

answers["EnKF_without_perturbations"] = ['MD', r'''
If $\Dobs = \bvec{0}$, then eqn. (A3) from the previous answer becomes
$$\begin{align}
    \barP
	&= (\I_{\\xDim}-\barK \bH)\barB(\I_{\\xDim}-\bH\tr \barK{}\tr) \tag{A5} \, ,
\end{align}$$
which shows that the updated covariance would be too small.
''']


answers['EnKF v1'] = ['MD', r'''
    def my_EnKF(N):
        E = mu0[:, None] + P0_chol @ rnd.randn(xDim, N)
        for k in range(1, nTime+1):
            # Forecast
            t   = k*dt
            E   = Dyn(E, t-dt, dt)
            E  += Q_chol @ rnd.randn(xDim, N)
            if k%dko == 0:
                # Analysis
                y        = yy[k//dko-1] # current obs
                Eo       = Obs(E, t)
                BH       = estimate_cross_cov(E, Eo)
                HBH      = estimate_mean_and_cov(Eo)[1]
                Perturb  = R_chol @ rnd.randn(p, N)
                KG       = divide_1st_by_2nd(BH, HBH+R)
                E       += KG @ (y[:, None] - Perturb - Eo)
            xa[k] = np.mean(E, axis=1)
''']

answers['rmse'] = ['MD', r'''
    rmses = np.sqrt(np.mean((xx-xa)**2, axis=1))
    average = np.mean(rmses)
''']

answers['Repeat experiment a'] = ['MD', r'''
 * (a). Set `p=1` above, and execute all cells below again.
''']

answers['Repeat experiment b'] = ['MD', r'''
 * (b). Insert `seed(i)` for some number `i` above the call to the EnKF or above the generation of the synthetic truth and obs.
''']

answers['Repeat experiment cd'] = ['MD', r'''
 * (c). Void.
 * (d). Use: `Perturb  = D_infl * R_chol @ rnd.randn(p, N)` in the EnKF algorithm.
''']


###########################################
# Tut: Benchmarking with DAPPER
###########################################
answers['jagged diagnostics'] = ['MD', r'''
Because they are only defined at analysis times, i.e. every `dko` time step.
''']

answers['RMSE hist'] = ['MD', r'''
 * The MSE will be (something close to) chi-square.
 * That the estimator and truth are independent, Gaussian random variables.
''']

answers['Rank hist'] = ['MD', r'''
 * U-shaped: Too confident
 * A-shaped: Too uncertain
 * Flat: well calibrated
''']

# Pointless...
# Have a look at the phase space trajectory output from `plot_3D_trajectory` above.
# The "butterfly" is contained within a certain box (limits for $x$, $y$ and $z$).
answers['RMSE vs inf error'] = ['MD', r'''
It follows from [the fact that](https://en.wikipedia.org/wiki/Lp_space#Relations_between_p-norms)
$ \newcommand{\x}{\x} \|\x\|_2 \leq {\\xDim}^{1/2} \|\x\|\_\infty \text{and}  \|\x\|_1 \leq {\\xDim}^{1/2} \|\x\|_2$
that
$$ 
\text{RMSE} 
= \frac{1}{K}\sum_k \text{RMSE}_k
\leq \| \text{RMSE}\_{0:k} \|\_\infty
$$
and
$$ \text{RMSE}_k = \| \text{Error}_k \|\_2 / \sqrt{{\\xDim}} \leq \| \text{Error}_k \|\_\infty$$
''']

answers['Twin Climatology'] = ['MD', r'''
    config = Climatology(**defaults)
    avergs = config.assimilate(HMM, xx, yy).average_in_time()
    print_averages(config, avergs, [], ['rmse_a', 'rmv_a'])
''']

answers['Twin Var3D'] = ['MD', r'''
    config = Var3D(**defaults)
    ...
''']
