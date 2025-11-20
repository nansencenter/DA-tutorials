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
# >>> show_answer('name')

from markdown import markdown as md2html # better than markdown2 ?
from IPython.display import HTML, display

from .macros import include_macros

def show_answer(tag, *subtags):
    """Display answer corresponding to 'tag' and any 'subtags'."""

    # Homogenize, e.g. 'a, b cd' --> 'abcd'
    subtags = ", ".join(subtags)
    subtags.translate(str.maketrans('', '', ' ,'))

    matching = list(filter(lambda key: key.startswith(tag), answers))
    if not matching:
        raise KeyError(f"No answer found for {tag=!r}")
    for key in matching:
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


def setup_typeset():
    """MathJax initialization for the current cell.

    This installs and configures MathJax for the current output.

    Necessary in Google Colab. Ref:
    https://github.com/googlecolab/colabtools/issues/322
    NB: this is NOT obsolete (Sept 2023)!
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
# Tut: Intro
###########################################
answers['state variables'] = ["MD", r"""
- (a) Position (x, y, z), velocity (vx, vy, vz).
- (b)
    - Epidemic (SEIR):  
        Susceptible (S), Exposed (E), Infectious (I), Recovered (R) populations.
    - Predator-prey (Lotka-Volterra):  
        Foxes (F), Rabbits (R).
- (c) Pressure, temperature, humidity, wind components (u, v, w), at each grid point.
- (d) Pressure, fluid saturations (oil, water, gas), possibly temperature, at each grid cell.
- (e) Concentrations of chemical species, temperature.
- (f) Vehicle density, average velocity, at each road segment.
- (g) Rating value(s) for each player/team.
- (h) Asset price, volatility.

General questions:

- Usually ordering does not affect the mathematics, but it might matters for implementation (e.g., mapping variables to indices).
- No, not for plain prediction purposes, but it is often convenient to have a fixed-length vector for computational efficiency and simplicity.
- It might increases computational cost/complexity and may introduce noise or bugs.
"""]

answers['model error'] = ["MD", r"""
Most models could be said to be afflicted by most of the shortcomings, but here are what appears most relevant:

- (a) Laws of motion and gravity: 1, 8
- (b) Epidemic (SEIR): 3, 5, 6, 7
- (c) Weather/climate forecasting: 3, 4, 7, 9
- (d) Petroleum reservoir flow: 4, 6, 7, 9
- (e) Chemical and biological kinetics: 3, 7
- (f) Traffic flow: 2, 4, 6, 10
- (g) Sports rating: 2, 7
- (h) Financial pricing: 2, 4, 7
"""]

answers['obs examples'] = ["MD", r"""
- (a) Altimetry, sextants, speedometers, compass readings, accelerometers, gyroscopes, or fuel-gauges
- (b)
    - SEIR: Number of positive tests, hospitalized, or dead. Sewage analyses. Google searches for symptoms.
    - Predator-prey: loss of harvest or poultry; direct (but incomplete) counts of foxes or rabbits,
        or their tracks, or their droppings.
- (c) Local weather stations. Satellite data. Weather balloons. Each of which can measure a range of variables,
  e.g. temperature, humidity, wind speed/direction, precipitation, radiances.
- (d) Well pressure, production rates (oil, gas, water), oil/water cuts.
- (e) (Bio)chemical analyses, spectrometry, temperature readings.
- (f) Vehicle counts, speed measurements, GPS traces, traffic cameras.
- (g) Game outcomes, scores, player statistics.
- (h) Asset prices, trading volumes, volatility indices.

"""]

answers['thesaurus 1'] = ["TXT", r"""
- Ensemble, Sample, Set of draws
- Stochastic, Random, Monte-Carlo
- Data, Measurements, Observations
"""]

answers['thesaurus 2'] = ["TXT", r"""
- Statistical inference, Inverse problems, Inversion, Estimation, Regression, Fitting
- Ensemble member, Sample point, Realization, Single draw, Particle
- Quantitative belief, Probability, Relative frequency, Estimate, Information, Uncertainty, Knowledge
- Recursive, Sequential, Iterative, Serial
- Function, Operator, Model, Transform(ation), Mapping, Relation
"""]

answers['Discussion topics 1'] = ['MD', r'''
 * (a) State estimation for large systems.
 * (b) "State variables" are (potentially unknown) variables that change in time.
   By contrast, "parameters" are constant-in-time (but potentially unknown) variables.
 * (c) "Prognostic" variables are *essential* for the prediction of the dynamical system.
   As such they are to be found among $x$ (contingent on the chosen parameterisation).
   By contrast, "diagnostic" variables can be derived (computed) from the prognostic/state variables.
   For example the observations, $y$, or other non-state quantities
   such as as momentum and energy (in case the state contains the velocity),
   or precipitation (in case the state contains pressure, humidity, salinity, ...).
 * (d) In priciniple, $t(k)$ is a point in time at which we receive a new observation.
       In practice, however, we tend to accumulate observations over some amount of time,
       and only *assimilate* them at regular intervals,
       determined by the frequency with which we *wish* to launch new forecasts.
       Note that this practice complicates the theory somewhat.
 * (e) In principle it's a science. In practice...
 * (f) Abstract concept to break the problem down into smaller, *recursive*, problems.  
   DAGs. Formalises the concept of hidden variables (states).
''']

###########################################
# Tut: Bayesian inference & Gaussians
###########################################
answers['pdf_G1'] = ['MD', r'''
    const = 1/np.sqrt(2*np.pi*sigma2)
    pdf_values = const * np.exp(-0.5*(x - mu)**2/sigma2)
''']

answers['Riemann sums a'] = ['MD', r'''
    # Midpoint rule, dx assumed constant
    dx = x[1] - x[0]
    mu = sum(f * x) * dx
    s2 = sum(f * (x-mu)**2) * dx

    # Alternatively: Right rule
    dxs = np.diff(x)
    mu = sum((f * x)[1:] * dxs)
    s2 = sum((f * (x-mu)**2)[1:] * dxs)
''']

answers['Riemann sums b'] = ['MD', r'''
    interval = abs(grid1d - mu) <= sigma
    prob = np.trapezoid(pdf_vals[interval], grid1d[interval])
    print(prob)
''']

answers['pdf_U1'] = ['MD', r'''
    height = 1/(b - a)
    pdf_values = height * np.ones_like(x)
    pdf_values[x<a] = 0
    pdf_values[x>b] = 0
    return pdf_values
''']

answers['Gauss integrals'] = ['MD', r'''
(i) $$\begin{align} \Expect[x]
&= \int x \, p(x) \,d x \tag{by definition} \\\
&= \int x \, c \, e^{-(x-\mu)^2 / 2 \sigma^2} \,d x \tag{by definition} \\\
&= \int (u + \mu) \, c \, e^{-u^2 / 2 \sigma^2} \,d u \tag{$u = x-\mu$}\\\
&= \int u \, c \, e^{-u^2 / 2 \sigma^2} \,d u
\;+\;  \mu \int \, c \, e^{-u^2 / 2 \sigma^2} \,d u \tag{distribute integral}\\\
&= \big[-\sigma^2 \, c \, e^{-u^2 / 2 \sigma^2}\big]^{+\infty}_{-\infty}
\;+\; \mu \, \Expect[1] \tag{integrate-by-parts + identify}
\end{align}
$$
The first term is zero. The second leaves only $\mu$, since $\Expect[1] = 1$.

(ii) $$\begin{align} \Expect[(x - \mu)^2]
&= \int (x - \mu)^2 \, c \, e^{-(x-\mu)^2 / 2 \sigma^2} \,d x \tag{by definition} \\\
&= \int u^2 \, c \, e^{-u^2 / 2 \sigma^2} \,d u \tag{$u = x-\mu$}\\\
&= \int u \, \big[ u \, c \, e^{-u^2 / 2 \sigma^2} \big] \,d u \tag{$u^2 = u\, u$} \\\
&= 0 - \int (1) \big[-\sigma^2 \, c \, e^{-u^2 / 2 \sigma^2}\big] \,d u \,,  \tag{integrate by parts} \\\
\end{align}
$$
where the first term was zero for the same reason as above,
and the second can again be expressed in terms of $\Expect[1] = 1$.
''']

answers['CVar in proba a'] = ['MD', r'''
[Link](https://stats.stackexchange.com/a/239594)
''']

answers['CVar in proba b'] = ['MD', r'''
The answer is in the link in the question,
more precisely [here](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician#Continuous_case).

But why is the result so intuitive?
Because the formal proof is a lot of ado for nothing;
it actually involves applying (integral) change-of-variables *twice*,
thereby cancelling itself out:

- Once to derive $p_u$ from $p_x$, although
  [differently](https://en.wikipedia.org/wiki/Integration_by_substitution#Application_in_probability)
  than in part (a).
- A second time when substituting $u$ by $\phi(x)$ in the integral for the expectation.
''']

answers['Sum of Gaussians a'] = ['MD', r'''
We could show this by letting $z = \phi(x) = \DynMod x + b$ and computing $\Expect z$
using $p_z(z) = p_x\big(\phi^{-1}(z)\big) \,/\, |\phi'(z)|$,
ref part (a) of [this question](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)-–-Change-of-variables).

But it is much easier to just apply part (b). Then
$\Expect [ \DynMod  x + b ] = \int ( \DynMod  x + b ) \, p(x) \,d x $,
from which the result follows from the linearity of the integral and the fact that $p(x)$ sums to 1.
''']

answers['Sum of Gaussians b'] = ['MD', r'''
$$\begin{align}
\mathbb{Var}[u] &= \Expect\Big[\big(u - \Expect(u)\big)^2 \Big] \\\
&= \Expect\Big[\big(\DynMod x + b - \Expect(\DynMod x + b) \big)^2 \Big] \\\
&= \Expect\Big[\DynMod^2 \big( x - \Expect(x) \big)^2 \Big] \\\
&= \DynMod^2 \Expect\Big[ \big( x - \Expect(x) \big)^2 \Big] \\\
&= \DynMod^2 \mathbb{Var}[x]\end{align}$$
''']


answers['Why Gaussian'] =  ['MD', r"""
What's not to love? Consider

 * The central limit theorem (CLT) and all its implications.
 * Pragmatism: Yields "least-squares problems", whose optima is given by a linear systems of equations.  
 * Self-conjugate: Gaussian prior and likelihood yields Gaussian posterior.
 * Among pdfs with independent components (2 or more),
   the Gaussian is uniquely (up to scaling) rotation-invariant (symmetric).
 * Gaussians have the maximal entropy for all distributions with a given variance.
 * Gaussians are invariant to self-convolution (i.e. the addition of two random variables)
 * The Gaussian is uniquely (among densities) invariant to the Fourier transform.
 * It is the heat kernel, i.e. the [Green's function](https://en.wikipedia.org/wiki/Green%27s_function#Table_of_Green's_functions)
   of the diffusion equation: one of the fundamental PDEs.
 * Uniquely, among elliptical distributions: uncorrelated, jointly distributed, normal random variables are independent.
 * Uniquely for Gaussian sampling distribution: maximizing the likelihood for the mean simply yields the sample average.
 * Unique in that the sample mean and variance are independent if calculated from a set of independent draws.
 * For more, see [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution#Properties)
   and Chapter 7 of: [Probability theory: the logic of science (Edwin T. Jaynes)](https://books.google.com/books/about/Probability_Theory.html?id=tTN4HuUNXjgC).
"""]

answers['Correlation extremes a'] = ['MD', r"""
Recall that $p(x, y) = p(x|y) \, p(y) = p(x) \, p(y)$ by independence.
Thus,
$$\begin{align}
\mathbb{Cov}(X, Y)
&= \int (x - \mu_x)(y - \mu_y) \, p(x, y) \, d x \, d y \\\
&= \Big(\int (x - \mu_x) p(x) \, d x \Big) \Big(\int (y - \mu_y) p(y) \, d y \Big)
\,.
\end{align}$$
But of course, $\int x \, p(x) \, d x = \mu_x$, so each integral is zero,
as is the resulting correlation.
"""]

answers['Correlation extremes b'] = ['MD', r"""
Now $p(y|x) = \delta(y - a x)$, where $\delta$ is the Dirac delta function
so that $\int f(y) \, p(y|x) \, d y = f(x)$ for any function $f$.
Hence
$$\begin{align}
\mathbb{Cov}(X, Y)
&= a \int (x - \mu_x)^2 \, p(x) \, d x \\\
&= a \sigma_x^2 \,.
\end{align}$$
Similarly, it can be shown that $\mathbb{Var}(Y) = a^2 \mathbb{Var}(X)$,
i.e. $\sigma_y = a \sigma_x$.
The conclusion follows by substitution into the definition of $\rho$.
"""]

answers['Correlation extremes c'] = ['MD', r"""
This is shown the same way as (b), except that
$a < 0$ implies $\sigma_y = -a \sigma_x$.
"""]

answers['RV linear algebra a'] = ['MD', r"""
- By the linearity of the integral,
  $$ \Expect[a X] = \int a x \, p(x) \, d x = a \int x \, p(x) \, d x = a \, \Expect[X] $$
- By the [law of the unconscious statistician](#Exc-(optional)-–-Change-of-variables)
  $$ \Expect[X + Y] = \iint (x + y) \, p(x, y) \, d x \, d y \,.$$
  Now consider $\iint x \, p(x, y) \, d x \, d y = \int x \, p(x) \Big(\int p(y | x) d y\Big) dx$.
  The inner integral is $1$ for any $x$ and so can be ignored, leaving just $\Expect[X]$.
  Likewise, $\iint y \, p(x, y) \, d x \, d y = \Expect[Y]$.
"""]

answers['RV linear algebra b'] = ['MD', r"""
$$\begin{align}
\mathbb{Var} [a X + Y]
&= \Expect\big[\big(a X + Y - \Expect[a X + Y]\big)^2\big] \\\
&= \Expect\big[\big(a (X - \Expect[X]) + (Y - \Expect[Y])\big)^2\big] \\\
&= \Expect\big[a^2 (X - \Expect[X])^2 + (Y - \Expect[Y])^2 + 2 a (X - \Expect[X])(Y - \Expect[Y])\big] \\\
\end{align}$$
The last term can be recognized as $2 a \, \mathbb{Cov}[X, Y]$,
which is zero by independence, as shown in the [correlation exercise](#Exc-–-correlation-extremes).
"""]

answers['RV linear algebra c'] = ['MD', r"""
Nothing really changes compared with part (b).
The whole exercise can be done element-wise.
"""]

answers['RV linear algebra e'] = ['MD', r"""
In the case of a diagonal covariance matrix,
the weight matrix on the norm in eqn. (GM) is also diagonal,
and therefore the norm reduces to a sum of the squares, $\sum_i z_i^2$.
Hence $p(\vect{z}) = \NormDist(\z | \vect{0}, \mat{I}) = c \prod_i e^{-z_i^2/ 2} = \NormDist(z | 0, 1) = \prod_i p(z_i)$.
"""]

answers['Broadcasting'] = ['MD', r"""
    # Cheating:
    # E = rng.multivariate_normal(mu, C, N).T

    # Using numpy "broadcasting"
    # (would have been a lot easier if using orientation N-times-d):
    # E = (mu + (L@Z).T).T

    # Make mu 2d:
    # mu2d = np.atleast_2d(mu).T
    mu2d = mu[:, None]
    # mu2d = np.tile(mu, (N, 1)).T
    # mu2d = np.outer(mu, np.ones(N))
    E = mu2d + L @ Z
"""]

answers['LG BR example'] = ['MD', r'''
- Eqn. (5) yields $P^\ta = \frac{1}{1/4 + 1/4} = \frac{1}{2/4} = 2$.
- Eqn. (6) yields $x^\ta = 2 \cdot (20/4 + 18/4) = \frac{20 + 18}{2} = 19$
''']

answers['symmetry of conjunction'] = ['MD', r'''
<a href="https://en.wikipedia.org/wiki/Bayes%27_theorem#For_continuous_random_variables" target="_blank">Wikipedia</a>

''']

answers["what's forward?"] = ['MD', r'''
Because estimation/inference/inverse problems
are looking for the "cause" from the "outcomes".
Of course, causality is a tricky notion.
Anyway, these problems tend to be harder,
generally because they involve solving (possibly approximately) a set of equations
rather than just computing a formula (corresponding to a forward problem).
Since $y = f(x)$ is common symbolism for formulae,
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
    - For the linear-Gaussian case, no.
- In the mixed cases: Yes. Otherwise: No.
- In fact, we'll see later that the linear-Gaussian posterior is Gaussian,
  and is therefore fully characterized by its mean and its variance.
  So we only need to compute its location and scale.
- The problem (of computing the posterior) is ill-posed:  
  The prior says there's zero probability of the truth being 
  in the region where the likelihood is not zero.
- This of course depends on the definition of "sufficient",
  but in the authors opinion $N \geq 40$ seems necessary.
''']

answers['Likelihood'] = ['MD', r'''
It follows from eqn. (Obs) that $p(y|x)$ is just $p(r)$ evaluated at $r = y - \ObsMod(x)$,  
i.e. $\NormDist(y - \ObsMod(x) | 0, R)$,  
i.e. $\NormDist(y | \ObsMod(x), R)$.

PS: for the non-additive case,
i.e. more complicated combinations of $r$ and $x$,
the likelihood can be derived by applying the
[change-of-variables formula](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)-–-Change-of-variables).
For example, multiplicative noise, i.e. $y = x r$, produces $p(y|x) = \NormDist(y | 0, x^2 R)$.
''']

answers['Observation models a'] = ['MD', r'''
The likelihood simply shifts/translates towards lower values by $15$.
''']
answers['Observation models b'] = ['MD', r'''
The distance from the origin to the likelihood (any point on it) gets halved.  
In addition, its width/spread gets halved, i.e. its precision doubles.

*PS*: the height of the likelihood remains unchanged.  
Thus, it no longer integrates to $1$. But this is not a problem,
since only densities need integrate to $1$;
the likelihood merely *updates* our beliefs.
''']
answers['Observation models c'] = ['MD', r'''
The likelihood now has 2 peaks, each one $\sqrt{y}$ away from $5$.
They're not actually quite Gaussian.

Note that a true inverse of $\ObsMod$ does not exist.
Yet BR gives us the next best thing: the two options, weighted.
A daring person might call it a "statistically generalized inverse".
''']
answers['Observation models d'] = ['MD', r'''
This is similar to the previous part of the exercise,
but the peaks are now centered on 0.

PS: the posterior is never quite fully Gaussian.
''']


answers['Multivariate Observations'] = ['MD', r'''
- $\bH = \begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix}$
- $\bH = $ N/A
- $\bH = \begin{bmatrix} 1 & 0 \end{bmatrix}$
- $\bH = \begin{bmatrix} 1 & \ldots & 1 \end{bmatrix}/D$
- $\bH = \begin{bmatrix} -1 & 1 \end{bmatrix}$
- $\bH = $ N/A
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

For comparison, there are about $10^{82}$ atoms in the universe.
''']

answers['BR Gauss, a.k.a. completing the square a'] = ['MD', r'''
Expanding the squares of the left hand side of eqn. (LG1),
and gathering terms in powers of $x$ yields
$$
    \frac{(x-x^\tf)^2}{P^\tf} + \frac{(x-y)^2}{R}
    =  x^2 (1/P^\tf + 1/R)
    - 2 x (x^\tf/P^\tf + y/R)
    + c_1
    \,, \tag{a1}
$$
with $c_1 = (x^\tf)^2/P^\tf + y^2/R$.
Now consider the right hand side of eqn. (LG1),
$$
    \frac{(x-x^\ta)^2}{P^\tf} + c_2
    = x^2 / P^\ta
    - 2 x x^\ta/P^\ta
    + (x^\ta)^2/P^\ta
    + c_2
    \,.
\tag{a2}
$$
where $c_2$ is constant wrt. $x$.
Both (a1) and (a2) are quadratics in $x$,
so we can equate them by requiring
$c_2 = c_1 - (x^\ta)^2/P^\ta$ and
$$ \begin{align}
1/P^\ta = 1/P^\tf + 1/R \,, \tag{a3} \\\
x^\ta/P^\ta = x^\tf/P^\tf + y/R \,, \tag{a4}
\end{align} $$
whereupon we immediately recover $P^\ta$ and $x^\ta$ of eqns. (5) and (6).

*PS: The above process is called "completing the square"
since it involves writing a quadratic polynomial as a single squared term
plus a "constant" that we add and subtract.*
''']

answers['BR Gauss, a.k.a. completing the square b'] = ['MD', r'''
$$ \begin{align}
p(x|y)
&\propto p(x) \, p(y|x) \\\
&=       N(x \mid x^\tf, P^\tf) \, N(y \mid x, R) \\\
&\propto \exp \Big( \frac{-1}{2} \big[ (x-x^\tf)^2/P^\tf + (x-y)^2/R \big] \Big) \,.
\end{align} \tag{} $$
The rest follows by eqn. (LG1) and identification with $N(x \mid x^\ta, P^\ta)$.
''']

answers['BR Gauss, a.k.a. completing the square c'] = ['MD', r'''
Resuming from part (a), we have
$$
    \frac{(x-x^\tf)^2}{P^\tf} + \frac{(x-y)^2}{R}
    =
    \frac{(x-x^\ta)^2}{P^\tf} + c_2
    \,,
\tag{a5}
$$
where
$$ \begin{align}
c_2
&= c_1 - (x^\ta)^2/P^\ta \\\
&= (x^\tf)^2/P^\tf + y^2/R - P^\ta (x^\tf/P^\tf + y/R)^2 \\\
&=
  y^2 ( 1/R - P^\ta/R^2 )
  - 2 y x^\tf \frac{P^\ta}{P^\tf R}
  + \frac{(x^\tf)^2}{P^\tf}
  - P^\ta \frac{(x^\tf)^2}{(P^\tf)^2}  
    \tag{a6}
\end{align} $$
Now, eqn. (a3) yields
$\frac{P^\ta}{P^\tf} = \frac{R}{P^\tf + R}$.
Similarly, $\frac{P^\ta}{R} = \frac{P^\tf}{P^\tf + R}$ and thereby
$ 1/R - P^\ta/R^2 = \frac{1}{R}(1 - \frac{P^\ta}{R} ) = \frac{1}{P^\tf + R} $.
Thus $c_2$ simplifies as
$$ \begin{align}
c_2
&= y^2 \frac{1}{P^\tf + R}  - 2 y x^\tf \frac{1}{P^\tf + R} + \frac{(x^\tf)^2}{P^\tf} - \frac{(x^\tf)^2}{P^\tf}\frac{R}{P^\tf + R} \\\
&= \frac{1}{P^\tf + R} \Bigl[ y^2 - 2 y x^\tf + (x^\tf)^2 \bigl( \frac{P^\tf + R}{P^\tf} - \frac{R}{P^\tf}\bigr ) \Bigr ] \\\
&= \frac{(y - x^\tf)^2}{P^\tf + R}
\tag{a7}
\end{align} $$
''']

answers['BR Gauss, a.k.a. completing the square d'] = ['MD', r'''
Let $U = X+Y$.
By the [law of the unconscious statistician](T2%20-%20Gaussian%20distribution.ipynb#Exc-(optional)-–-Change-of-variables)
$$ p_u(u) = \int p_x(x) \, p_q(u - x) \, d x \,. \tag{}$$

Insert eqn. (G1) for the Gaussian pdf.
Then, the products of the exponentials of the integrand yield an exponent that is a sum of squares,
which we can rewrite using part (c)
to obtain a quadratic in $x$ and another in $u$.
The latter factors out of the integral,
rendering the remaining integral a constant (in $u$).
The resulting Guassian in $u$ has the parameters are the ones we are looking for!
''']

answers['BR Gauss'] = ['MD', r'''
We can ignore factors that do not depend on $x$.

\begin{align}
p(x|y)
&= \frac{p(x) \, p(y|x)}{p(y)} \\\
&\propto p(x) \, p(y|x) \\\
&=       N(x \mid x^\tf, P^\tf) \, N(y \mid x, R) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (x-x^\tf)^2/P^\tf + (x-y)^2/R \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (1/P^\tf + 1/R)x^2 - 2(x^\tf/P^\tf + y/R)x \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( x - \frac{x^\tf/P^\tf + y/R}{1/P^\tf + 1/R} \Big)^2 \cdot (1/P^\tf + 1/R) \Big) \,.
\end{align}

Identifying the last line with $N(x \mid x^\ta, P^\ta)$ yields eqns (5) and (6).
''']

answers['BR Kalman1 algebra'] = ['MD', r'''
- Multiplying eqn. (5) by $1 = \frac{P^\tf R}{P^\tf R}$ yields
  $P^\ta = \frac{P^\tf R}{P^\tf + R} = \frac{P^\tf}{P^\tf + R} R$, i.e. eqn. (8).
- We also get $P^\ta = \frac{R}{P^\tf + R} P^\tf$.
  Adding $0 = P^\tf - P^\tf$ to the numerator yields eqn. (10).
- Applying formulae (8) and (10) for $P^\ta$
  in eqn. (6) immediately produces eqn. (11).
''']

answers['KG intuition'] = ['MD', r'''
Consider eqn. (9). Both nominator and denominator are strictly larger than $0$, hence $K > 0$.
Meanwhile, note that $K$ weights the observation uncertainty $(R)$
vs. the total uncertainty $(P^\tf + R)$, whence $P^\tf + R > P^\tf$, i.e. $K<1$.

Since $0<K<1$, eqn. (8) yields $P^\ta < R$,
while eqn. (10) yields $P^\ta < P^\tf$.

From eqn. (11), $x^\ta = (1-K) x^\tf + K y$.
Since $0<K<1$, we can see that $x^\ta$
is a 'convex combination' or 'weighted average'.
*For even more detail, consider the case $x^\tf<y$ and then case $y<x^\tf$.*

Because it describes how much the esimate is "dragged" from $x^\tf$ "towards" $y$.  
I.e. it is a multiplification (amplification) factor,
which French (signal processing) people like to call "gain".  
''']

answers['BR Kalman1 code'] = ['MD', r'''
    KG = H*Pf / (H**2*Pf + R)
    Pa = (1 - KG*H) * Pf
    xa = xf + KG * (y - H*xf)
''']

answers['Posterior cov'] =  ['MD', r"""
  * No.
      * It means that information is always gained.
      * No, not always.  
        But on average, yes:  
        the "expected" posterior entropy (and variance)
        is always smaller than that of the prior.
        This follows from the [law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance).
  * It probably won't have decreased. Maybe you will just discard the new information entirely, in which case your certainty will remain the same.  
    I.e. humans are capable of thinking hierarchically, which effectively leads to other distributions than the Gaussian one.
"""]

answers['MMSE'] =  ['MD', r"""
Inserting $0 = \mu - \mu$ into the expression for the MSE decomposes it into the squared bias (if any)
plus the variance (of $X$ itself), which is independent of the choice of point estimate.
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

answers['RV sums'] = ['MD', r'''
By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
and that of (Dyn),
the mean parameter becomes:
$$ \\Expect(\\DynMod x+q) =  \\DynMod \\Expect(x) + \\Expect(q) = \\DynMod x^\ta \,. $$

Moreover, by independence,
$ \\text{Var}(\\DynMod x+q) = \\text{Var}(\\DynMod x) + \\text{Var}(q) $,
and so
the variance parameter becomes:
$$ \\text{Var}(\\DynMod x+q) = \\DynMod^2 P^\ta + Q \,.  $$
''']

answers['LinReg deriv a'] = ['MD', r'''
$$ \begin{align}
p\, (y_1, \ldots, y_K \;|\; a)
&= \prod_{k=1}^K \, p\, (y_k \;|\; a) \tag{each obs. is indep. of others, knowing $a$.} \\\
&= \prod_k \, \NormDist{N}(y_k \mid a k,R) \tag{inserted eqn. (3) and then (1).} \\\
&= \prod_k \,  (2 \pi R)^{-1/2} e^{-(y_k - a k)^2/2 R} \tag{inserted eqn. (G1) from T2.} \\\
&= c \exp\Big(\frac{-1}{2 R}\sum_k (y_k - a k)^2\Big) \\\
\end{align} $$
Taking the logarithm is a monotonic transformation, so it does not change the location of the maximum location.
Neither does diving by $c$. Multiplying by $-2 R$ means that the maximum becomes the minimum.
''']

answers['x_KF == x_LinReg'] = ['MD', r'''
We'll proceed by induction.  

With $P^\tf_1 = \infty$, we get $P^\ta_1 = R$,
which initializes (13).  

Now, inserting (13) in (12) yields:

$$
\begin{align}
P^\ta_{K+1} &= 1\Big/\big(1/R + \textstyle (\frac{K}{K+1})^2 / P^\ta_K\big)
\\\
&= R\Big/\big(1 + \textstyle (\frac{K}{K+1})^2 \frac{\sum_{k=1}^K k^2}{K^2}\big)
\\\
&= R\Big/\big(1 + \textstyle \frac{\sum_{k=1}^K k^2}{(K+1)^2}\big)
\\\
&= R(K+1)^2\Big/\big((K+1)^2 + \sum_{k=1}^K k^2\big)
\\\
&= R(K+1)^2\Big/\sum_{k=1}^{K+1} k^2
\,,
\end{align}
$$
which concludes the induction.

The proof for $x^\ta_k$ is similar.
''']


answers['KF1 code'] = ['MD', r'''

            # Forecast step
            xf = M * xa
            Pf = M**2 * Pa + Q
            # Analysis update step
            Pa = 1 / (1/Pf + H**2/R)
            xa = Pa * (xf/Pf + H*obsrvs[k]/R)
            # Alternatively:
            # KG = H*Pf / (H**2*Pf + R)
            # Pa = (1 - KG*H) * Pf
            # xa = xf + KG * (y - H*xf)

''']

answers['Asymptotic Riccati a'] = ['MD', r'''
Follows directly from eqn. (6) from both this tutorial and
[the previous one](T3%20-%20Bayesian%20inference.ipynb#Exc-–-BR-LG1).
''']

answers['Asymptotic Riccati b'] = ['MD', r'''
From the previous part of the exercise,
if $\DynMod = 1$ then
$1/P^\ta_k = 1/P^\ta_{k-1} + 1/R$
and so the difference $1/P^\ta_k - 1/P^\ta_{k-1}$
is always $1/R$.
In other words, $1/P^\ta_k$ grows linearly with $1/R$,
starting from $1/P^\ta_0$.
''']

answers['Asymptotic Riccati c'] = ['MD', r'''
From the previous part of the exercise,
$1/P^\ta_k \xrightarrow[k \rightarrow \infty]{} +\infty$
''']

answers['Asymptotic Riccati d'] = ['MD', r'''
Since $1 / \DynMod^2 > 1$,
$1/P^\ta_k$ now grows quicker than
in the previous parts of the exercise,
whence the result.

Note that since this is just an inequality,
it holds also for time-dependent $\DynMod_k$
as long as they're less than 1.
''']

answers['Asymptotic Riccati e'] = ['MD', r'''
The fixed point $P^\ta_\infty$ should satisfy
$P^\ta_\infty = 1/\big(1/R + 1/[\\DynMod^2 P^\ta_\infty]\big)$,
yielding the answer.

Note that this asymptote is **not 0**!
In other words, even though the KF keeps gaining observational data/information,
this gets balanced out by the growth in error/uncertainty during the forecast.
Also note that the asymptotic state uncertainty ($P^\ta_\infty$)
is directly proportional to the observation uncertainty ($R$).
''']

answers['Asymptotes when Q>0 a'] = ['MD', r'''
Again by merging the forecast and analysis steps, we get
$
1/P^a = 1/Q + 1/R
$
which immediately yields
$P^a = 1/ (1/Q + 1/R)$.
''']

answers['Asymptotes when Q>0 b'] = ['MD', r'''
Multiply
$$
1/P^a = \frac{1}{P^a + Q} + 1/R
$$
by all of the denominators.
Cancel $P^a R$ from both sides.
''']

answers['Asymptotes when Q>0 c'] = ['MD', r'''
As the terms including $Q$ come to dominate,
the equation tends to $P = R - \varepsilon$
with $\varepsilon \rightarrow 0^+$.

This means that if the model error is practically infinite,
then previous information is worthless,
and the uncertainty after each analysis step
equals that of the single observation.
''']

answers['Asymptotes when Q>0 d'] = ['MD', r'''
If $Q \rightarrow 0$
then $P \rightarrow 0$,
so the middle term $Q P$ can be neglected,
leaving $P^2 = Q R$.

Thus, $P$ tends to zero if the model error vanishes
(as we also found previously),
but also note that $\sqrt{Q R}$ is perfectly symmetric in $R$ and $Q$.
Indeed, we'd find the same asymptote if we let $R \rightarrow +\infty$.
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

answers['KF precision'] = ['MD', r'''
By Bayes' rule:
$$\begin{align}
- 2 \log p(\x|\y) =
\|\ObsMod \x-\y \|\_\R^2 + \| \x - \x^\tf \|\_{\bP^\tf}^2
 + \text{const}_1
\,.
\end{align}$$
Expanding, and gathering terms of equal powers in $\x$ yields:
$$\begin{align}
- 2 \log p(\x|\y)
&=
\x\tr \left( \ObsMod\tr \Ri \ObsMod + (\bP^\tf)^{-1}  \right)\x
- 2\x\tr \left[\ObsMod\tr \Ri \y + (\bP^\tf)^{-1} \x^\tf\right] + \text{const}_2
\, .
\end{align}$$
Meanwhile
$$\begin{align}
\| \x-\x^\ta \|\_{\bP^\ta}^2
&=
\x\tr (\bP^\ta)^{-1} \x - 2 \x\tr (\bP^\ta)^{-1} \x^\ta + \text{const}_3
\, .
\end{align}$$
Eqns (5) and (6) follow by identification.
''']

answers['nD-covars are big a'] = ['MD', r'''
${\\xDim}$-by-${\\xDim}$
''']
answers['nD-covars are big b'] = ['MD', r'''
$2 {\\xDim}^3/3$.
''']
answers['nD-covars are big c'] = ['MD', r'''
Assume ${\bP^\tf}$ stored as float (double). Then it's 8 bytes/element.
And the number of elements in ${\bP^\tf}$: ${\\xDim}^2$. So the total memory is $8 {\\xDim}^2$.
''']
answers['nD-covars are big d'] = ['MD', r'''
8 trillion bytes. I.e. 8 million MB.
''']
answers['nD-covars are big e'] = ['MD', r'''
$(2^3)^2 = 64$ times more.

*PS: In addition, note that the CFL condition may require
you to forecast the dynamics $2^o$ times slower,
where $o$ is the highest order of temporal derivatives in your PDEs.*
''']

answers['Woodbury general'] = ['MD', r'''
We show that the left hand side (LHS) -- without the surrounding inverse --
gets "cancelled" by multiplication with the RHS.
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
by replacing $\V, \U$ by $\ObsMod$,
*provided that everything is still well-defined*.
In other words,
we need to show the existence of the left hand side.

Now, for all $\x \in \Reals^{\\xDim}$, $\x\tr \B^{-1} \x > 0$ (since $\B$ is SPD).
Similarly, $\x\tr \ObsMod\tr \R^{-1} \ObsMod \x\geq 0$,
implying that the left hand side is SPD:
$\x\tr (\ObsMod\tr \R^{-1} \ObsMod + \B^{-1})\x > 0$,
and hence invertible.
''']

answers['Woodbury C2'] = ['MD', r'''
A straightforward validation of (C2)
is obtained by cancelling out one side with the other.
A more satisfying exercise is to derive it from (C1)
starting by right-multiplying by $\ObsMod\tr$.
''']


###########################################
# Tut: Dynamical systems, chaos, Lorenz
###########################################

answers["rk4"] = ["MD", r'''

    ens = initial_states
    integrated.append(ens)
    for k, t in enumerate(time_steps[:-1]):
        ens = rk4(dxdt_fixed, ens.T, t, dt).T
        integrated.append(ens)
    return np.swapaxes(integrated, 0, 1), time_steps

Note that such double transposing is not the only way to vectorise.
It is often better to do [something else](https://nansencenter.github.io/DAPPER/reference/mods/).
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

answers["error dynamics a"] = ["MD", r"""
$\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
"""]
answers["error dynamics b"] = ["MD", r"""
Differentiate $e^{F t}$.
"""]
answers["error dynamics c"] = ["MD", r"""
- (1) Dissipates to 0.
- (2) No.
  A balance is always reached between
  the uncertainty reduction $(1-K)$ and growth $F^2$.  
"""]
answers["error dynamics d"] = ["MD", r"""
Since we want
$\varepsilon(t) = 2 \varepsilon_0$
we need $2 = e^{F t}$, i.e. $t = \log(2) / F$.
"""]
answers["error dynamics e"] = ["MD", r"""
It models (i.e. emulate) asymptotic ($t \to \infty$) saturation
of the error at $\varepsilon_\infty$, rather than indefinite exponential growth.
Ref [Wikipedia](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)

Solution: $\varepsilon(t) = \frac{\varepsilon_{\infty}}{1 + e^{F \, t}}$.
"""]
answers["error dynamics f"] = ["MD", r"""
$\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
with $f$ and $g$ both evaluated in $x$.

Phenomenon: initial, linear (rather than exponential) growth.
"""]
answers["error dynamics g"] = ["MD", r"""
Substitute $\vect{u} = \mat{T}^{-1} \vect{x}$ into eqn. (TLM).
"""]

answers["Bifurcations63 a"] = ["MD", r"""
The origin, i.e. $(0, 0, 0)$
"""]

answers["Bifurcations63 b"] = ["MD", r"""
At $\rho=1$.
To detect it visually, zoom in and increase `Time`.
"""]

answers["Bifurcations63 c"] = ["MD", r"""
At $\rho = 1.3456$.
However, it's can be hard to visually detect for $\rho < 10$.
"""]

answers["Bifurcations63 d"] = ["MD", r"""
The difference is that for $\rho < 13.926$
no transition between the lobes/wings takes place
(apart from trajectories that are not yet on the attractor).

For larger $\rho$ this is possible, though not frequent.
"""]

answers["Bifurcations63 e"] = ["MD", r"""
Chaos requires $\rho > 24.74$.
It is visible in 3D by the hollowing out of the centres of the lobes.
It is more clear in 2D: there is no longer any convergence
(even for very long time integration).
"""]

answers["Bifurcations63 f"] = ["MD", r"""
It is periodic.
"""]

answers["Bifurcations63 f"] = ["MD", r"""

Setting eqn. (1) to zero immediately yields
$y = x$
and
$$\\left\\{\begin{align}
    0 &= x (\rho - 1 - z)  \,, \\\
    \beta z &= x^2 \,,
\end{align}\\right.$$
which has solution $x = z = 0$
or
$$\\left\\{\begin{align}
z&=\rho - 1 \,, \\\
x&=\pm \sqrt{\beta z}
\end{align}\\right.$$
which only exists if $\rho>1$.

For more mathematical analysis,
in particular the stability of these stable points
(defined by the characteristic polynomials of the Jacobian)
see [here](https://web.math.ucsb.edu/~jhateley/paper/lorenz.pdf).
"""]

answers['Guesstimate 63'] = ["MD", r"""
It is arguably around 0.7, since
the largest Lyapunov exponent is 0.9.

However, as discussed by [Anderson and Hubeny 1997](https://www.gfdl.noaa.gov/bibliography/related_files/jla9702.pdf),
there are several ways to define the doubling time, with differing numerical answers.
"""]

answers["Bifurcations96 a"] = ["MD", r"""
$F<0.895$.
"""]

answers["Bifurcations96 b"] = ["MD", r"""
$F<4$.
"""]

answers["Bifurcations96 c"] = ["MD", r"""
$F \geq 4$.
"""]

answers["doubling time"] = ["MD", r"""
    ens     = trajectories96[:, -1]    # Ensemble of particles at the end of integration
    vr      = np.var(ens, axis=0)      # Variance (spread^2) of final ensemble
    vr      = np.mean(vr)              # Homogenize
    spread  = np.sqrt(vr)              # Std. dev.
    eps     = [FILL IN SLIDER VALUE]   # Initial spread
    Time    = [FILL IN SLIDER VALUE]   # Integration time
    lyap    = np.log(spread/eps)/Time  # Assumes `spread = eps * exp(lyap * Time)`
    print("Doubling time (approx):", np.log(2)/lyap)
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

answers['Simple kriging'] = ["MD", r"""
    covar_yy = C0 - vg(dists_yy)
    cross_xy = C0 - vg(dists_xy)
    # weights = sla.inv(covar_yy) @ cross_xy.T
    # weights = sla.solve(covar_yy, cross_xy.T)
    weights, *_ = sla.lstsq(covar_yy, cross_xy.T, cond=1e-9)
"""]

answers['Interpolant = f(Variogram) a'] = ["MD", r"""
When `Range` $\to 0$ then the SK (mean) estimate becomes a flat line
located on the supposed mean (`mu`),
except where there are observations, which gets interpolated by a spike.
In other words, it becomes a collection (sum) of delta functions.
The same effect can be achieved regardless of variogram model by setting the nugget to 1.

When `Range` $\to \infty$ then the SK field estimate becomes a (piecewise) linear interpolant.
The extrapolating tails appear to tend towards `mu`.

For intermediate values, the SK interpolant resembles somewhat the covariance function in between the data points. It looks like a slack curtain draped over a set of spikes corresponding to the observations.
"""]

answers['Interpolant = f(Variogram) b'] = ["MD", r"""
The interpolant is now smooth. Thus, even as `Range` $\to \infty$ the interpolant never becomes piecewise linear. Instead, it will curve and bend a lot around the observational data, which also has the effect that it can exceed the range of the observations. In fact, the interpolant is infinitely differentiable, meaning analytic, meaning that knowing it should be possible to extrapolate the entire function based on observations located within a local neighbourhood, which is somewhat against the spirit of a random function.
"""]

answers['Ordinary kriging a'] = ["MD", r"""
    def ordinary_kriging(vg, dists_xy, dists_yy, observations):
        xDim = len(dists_xy)
        yDim = len(dists_yy)
        A = np.ones((yDim+1, yDim+1))
        b = np.ones((yDim+1, xDim))

        A[-1, -1] = 0
        A[:-1, :-1] = vg(dists_yy)
        b[:-1] = vg(dists_xy).T

        weights, *_ = sla.lstsq(A, b, cond=1e-9)
        return observations @ weights[:-1]
"""]
answers['Ordinary kriging b'] = ["MD", r"""
`transf01=none`.
"""]
answers['Ordinary kriging c'] = ["MD", r"""
Let `linear`, `quadratic`, `cubic` refer to `power=1,2,3`.

- The `linear` variogram yields a piecewise linear intepolant.
  Unlike the `triangular` one (which can be used with SK)
  there are no longer any weird vertices at `Range` distance from observations.
- The `quadratic` variogram always yields a single/global quadratic,
  losing the ability to interpolate.
  Even just slightly perturbing the exponent (from $2$ to $2.1$, e.g., or $1.9$)
  recovers the interpolating behavioour.
- The `cubic` one yields an interpolant that extrapolates towards unbounded values,
  rather than attenuating towards any mean.
"""]
answers['Ordinary kriging d'] = ["MD", r"""
The SK interpolant tends to `mu` away from the observation, while the OK interpolant is constant/flat line with value equal to the observation.
"""]
answers['Ordinary kriging e'] = ["MD", r"""
The SK interpolant tends to `mu` away from the observation, while the OK interpolant tends towards the average of the 2 observations.
"""]

answers['Universal kriging a'] = ["MD", r"""
    def universal_kriging(vg, dists_xy, dists_yy, observations, regressors, regressands):
        xDim = len(dists_xy)
        yDim = len(dists_yy)

        A = np.zeros((yDim+2, yDim+2))
        b = np.zeros((yDim+2, xDim))

        A[:-2, :-2] = vg(dists_yy)
        A[-2:,:-2] = regressors
        A.T[-2:,:-2] = regressors

        b[:-2] = vg(dists_xy).T
        b[-2:] = regressands

        weights, *_ = sla.lstsq(A, b, cond=1e-9)
        return observations @ weights[:-2]
"""]
answers['Universal kriging b'] = ["MD", r"""
    from scipy.interpolate import CubicSpline
    ax.plot(grid, CubicSpline(obs_loc, observs, bc_type="natural")(grid), 'C4', label="N-spline")

    # from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
    # ax.plot(grid, Akima1DInterpolator(obs_loc, observs)(grid), 'C6', label="Akima spline")
    # ax.plot(grid, PchipInterpolator(obs_loc, observs)(grid), 'C5', label="PCHIP spline")

Yes, by using a cubic variogram.
While the extrapolating tails differ, the interpolant lines are superimposed in the interior.

This is an interesting results because cubic (piecewise) splines are
named after shipbuilding techniques and can be similarly shown to minimize
bend, and their derivation typically proceeds from requiring continuity of the
second order from one cubic to another, which is quite different to the
derivation of kriging. Moreover, their implementation is also very different,
with the cubic spline computed from a tridiagonal linear system of equations.
"""]
answers['Universal kriging c'] = ["MD", r"""
Use only 2 or 3 observation points. To make it clearer yet, add a linear trend to the truth.
"""]

answers['variogram params'] = ["MD", r"""
- (a) They become flat (fully correlated infinitely far apart). Effectively each field realisation becomes a single draw.
- (b) Covariance becomes the identity. Fields become white noise.
- (c) Yes, set `nugget = 1`.
- (d) Because the Gaussian variogram is differentiable at zero
  (variograms apply to *distances*, and as such are applied for $-\epsilon$ as well as $+\epsilon$).
  The behaviour of the variogram near the origin determines the smoothness at small scales.
- (e) It controls the "vertical" spacing between fields.
  This is because the sill equals $C(0)$, i.e. the "variance" of the field (assuming this exists).
"""]



###########################################
# Tut: Ensemble representation
###########################################

answers['KDE'] = ['MD', r'''
    from scipy.stats import gaussian_kde
    ax.plot(grid1d, gaussian_kde(E.ravel(), 10.**log_bw)(grid1d), label="KDE estimate")
''']

answers['ensemble moments, loop'] = ['MD', r'''
    xDim, N = E.shape
    x_bar = np.zeros(xDim)
    C_bar = np.zeros((xDim, xDim))

    for n in range(N):
        x_bar += E[:, n]
    x_bar /= N

    for n in range(N):
        xc = (E[:, n] - x_bar)[:, None]  # "x_centered"
        C_bar += xc @ xc.T
        # C_bar += np.outer(xc, xc)
    C_bar /= (N-1)
''']

answers['variance estimate statistics'] = ['MD', r'''
 * Visibly, the expected value (mean) of $1/\barC$ is not $1$,
   so $1/\barC$ is not unbiased. This is to be expected,
   since taking the reciprocal is a *nonlinear* operation.

   In the EnKF, estimated covariances feature in multiple places,
   interacting and impacting the updated ensemble nonlinearly.
   Thus it is not obvious that unbiasedness is all that important
   for covariance estimation for the EnKF.
 * The mean of $\barC$ is $1$ for any ensemble size.
 * The mean  of $1/\barC$ is infinite for $ N=2 $,
   and decreases monotonically as $ N $ increases,
   tending to $1$ as $ N $ tends to $+\infty$.

''']


answers['ensemble moments vectorized a'] = ['MD', r'''
Show that 

- $\E \ones / N = \bx$, and
- $\bx \ones^T = \begin{bmatrix} \bx, & \ldots & \bx \end{bmatrix} \,.$  
''']

answers['ensemble moments vectorized b'] = ['MD', r'''
Show that element $(i, j)$ of the matrix product $\X^{} \Y^T$  
equals element $(i, j)$ of the sum of the outer product of their columns:
$\sum_n \x_n \y_n^T$.  
Put this in the context of $\barC$.
''']

answers['ensemble moments vectorized c'] = ['MD', r'''
On the one hand, note one such Cholesky factor is the (centered) ensemble matrix itself, $\mat{X}$,
divided by $\sqrt{N{-}1}$. Thus the ensemble characterises/represents a covariance matrix (estimate) and
– by extension (Gaussianity assumptions) – a particular distribution.

On the other hand, because there are (infinitely) many such Cholesky factors that would fit,
each separated from one another by a $\xDim$ rotation matrix applied on the right,
since any unitary $\mat{\Omega}$ matrix satisfies $\mat{\Omega} \mat{\Omega}\tr = \mat{I}_{\xDim}$.

As elaborated by [Sakov (2008)](#References),
the non-uniqueness reflects the fact that many different ensembles (think scatter plots)
will have the same estimated covariance matrix, i.e. represent the same distribution.
In the context of an ensemble _update_,
discriminating between different ensembles with the same covariance
leads to the theory of **optimal transport**;
an important result is that the (unique) symmetric Cholesky factor is the one closest to the identity matrix,
i.e. yielding the smallest transport, in some metric [(Ott, 2004)](#References).
''']

answers['ensemble moments vectorized e'] = ['MD', r'''
$\X$ only has $\xDim \times N$ elements, while $C$ has $xDim \times \xDim$ elements.
''']

answers['ensemble moments vectorized e'] = ['MD', r'''
Use the following code:

    x_bar = np.sum(E, axis=1, keepdims=True)/N
    X     = E - x_bar
    C_bar = X @ X.T / (N-1)   
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
 * (a). Error: discrepancy between estimator and truth.
   Residual: discrepancy between explained and observed data.
 * (b). Bias = *average* (i.e. systematic) error.
        The remainder, being closer to unbiased "noise", is often termed **sampling error**.
 * (c). [Wiki](https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship)
''']

answers['associativity'] = ['MD', r'''
Let $X = \E^\tf \AN$ be the centered ensemble,
where $\AN$ was defined above,
and let $Y = \ObsMod \, \E^\tf \AN$.
Then
$ \ObsMod \, \barC_\x^\tf
= \ObsMod \, \X \X\tr
= \Y \X\tr \,,$
which can be identified as the cross covariance.

''']

###########################################
# Tut: Writing your own EnKF
###########################################
answers["EnKF_nobias_a"] = ['MD', r'''
Let $\ones$ be the vector of ones of length $N$. Then
$$\begin{align}
    \bx^\ta
    &= \frac{1}{N} \E^\ta \ones \tag{because $\sum_{n=1}^N \x^\ta_n = \E^\ta \ones$.} \\\
    &= \frac{1}{N} \E^\tf \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Dobs - \bH \E^\tf \right) \ones \tag{inserting eqn. (4).}
\end{align}$$
Assuming $\Dobs \ones=\\vect{0}$ yields eqn. (6).
One might say that the mean of the EnKF update conforms to the KF mean update.  

"Conforming" is not a well-defined math word.
However, the above expression makes it clear that $\bx^\ta$ is linear with respect to $\Dobs$, so that
$$\begin{align}
    \Expect \bx^\ta
    &= \frac{1}{N} \E^\tf \ones + \frac{1}{N} \barK
    \left(\y\ones\tr - \Expect\Dobs - \bH \E^\tf \right) \, .
\end{align}$$

Now, since $\Expect \r_n = \vect{0}$, it follows that $\Expect \Dobs = \vect{0}$,
and we recover eqn. (6).

The conclusion: the mean EnKF update is unbiased...
However, this is only when $\E^\tf$ is considered fixed, and its moments assumed correct.
''']

answers["EnKF_nobias_b"] = ['MD', r'''
First, compute the updated anomalies, $\x^\ta$, by inserting  eqn. (4) for $\E^\ta$:
$$\begin{align}
	\x^\ta
	&= \E^\ta \AN \\\
	%&= {\X} + \barK\left(\y \ones\tr - \D - \bH \E^\tf\right) \AN \\\
	&= {\X} - \barK \left[\D + \bH \X\right] \,, \tag{A1}
\end{align}$$
where the definition of $\D$ has been used.

Inserting eqn. (A1) for the updated ensemble covariance matrix, eqn. (7b):
$$\begin{align}
	\barP
	&= \frac{1}{N-1} \x^\ta{\x^\ta}\tr \\\
    %
	&= \frac{1}{N-1} \left({\X} - \barK\left[\D + \bH \X\right]\right)
	\left({\X} - \barK\left[\D + \bH \X\right]\right)\tr \, .  \tag{A2}
\end{align}$$

Writing this out, and employing
the definition of $\barP^\tf$, eqn. (7a), yields:
$$\begin{align}
    \barP
	&= \barP^\tf + \barK \bH \barP^\tf \bH\tr \barK{}\tr -  \barP^\tf \bH\tr \barK{}\tr - \barK \bH \barP^\tf \tag{A3} \\\
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
	&=  \barP^\tf  + \barK \bH \barP^\tf \bH\tr \barK{}\tr -  \barP^\tf \bH\tr \barK{}\tr - \barK \bH \barP^\tf  + \barK \R \barK{}\tr \tag{A4} \\\
	&=  (\I_{\\xDim} - \barK \bH) \barP^\tf + \barK(\bH \barP^\tf \bH\tr + \R)\barK{}\tr -  \barP^\tf \bH\tr \barK{}\tr \tag{regrouped.} \\\
	&=  (\I_{\\xDim} - \barK \bH) \barP^\tf + \barP^\tf \bH\tr \barK{}\tr -  \barP^\tf \bH\tr \barK{}\tr \,, \tag{inserted eqn. (5a).} \\\
    &=  (\I_{\\xDim} - \barK \bH) \barP^\tf \,. \tag{10}
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
If $\Dobs = \vect{0}$, then eqn. (A3) from the previous answer becomes
$$\begin{align}
    \barP
	&= (\I_{\\xDim}-\barK \bH)\barP^\tf(\I_{\\xDim}-\bH\tr \barK{}\tr) \tag{A5} \,,
\end{align}$$
which shows that the updated covariance would be too small.
''']


answers['EnKF v1'] = ['MD', r'''

    def my_EnKF(N):
        """My implementation of the EnKF."""
        ### Init ###
        E = xa[:, None] + Pa12 @ rnd.randn(xDim, N)
        for k in tqdm(range(1, nTime+1)):
            t = k*dt
            ### Forecast ##
            E = Dyn(E, t-dt, dt)
            E += Q12 @ rnd.randn(xDim, N)
            if k % dko == 0:
                ### Analysis ##
                y = obsrvs[[k//dko-1]].T  # current observation
                Eo = Obs(E, t)            # observed ensemble
                # Compute ensemble moments
                Y = Eo - Eo.mean(keepdims=True)
                X = E - E.mean(keepdims=True)
                PH = X @ Y.T / (N-1)
                HPH = Y @ Y.T / (N-1)
                # Compute Kalman Gain
                KG = nla.solve(HPH + R, PH.T).T
                # Generate perturbations
                Perturbs = R12 @ rnd.randn(p, N)
                # Update ensemble with KG
                E += KG @ (y - Eo - Perturbs)
            # Save statistics
            ens_means[k] = np.mean(E, axis=1)
            ens_vrncs[k] = np.var(E, axis=1, ddof=1)

''']

answers['rmse'] = ['MD', r'''
    rmses = np.sqrt(np.mean((truth - estimates)**2, axis=1))
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
