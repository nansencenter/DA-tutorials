# Beyond Colab: evaluating a replacement platform for DA-tuts

*Status as of 2026-08-19. Living document — see `pilot/marimo/` for the working pilot and `pilot/jupyterbook/` for bare scaffolding of the other pilot.*

## Purpose

DA-tuts (`nansencenter/DA-tutorials`) is a 9-notebook (T1–T9) data-assimilation
tutorial series, historically distributed via Google Colab. The goal of this
work is a **single-click-runnable set of tutorial notebooks for students**:
open a link, get a working, editable, re-runnable notebook — no install, no
account required beyond what's unavoidable, no "it worked yesterday" surprises.

## Why Colab is being replaced

Colab itself is the dominant source of friction, independent of anything
specific to how DA-tuts is authored:

- **Frozen frontend** — Colab's own UI/runtime hasn't meaningfully evolved,
  while the rest of the notebook ecosystem has.
- **Drifting base-image dependencies** — the preinstalled package set changes
  under students' feet between semesters, silently breaking cells that worked
  before.
- **No anchor-link resolution** — in-notebook cross-references (e.g. "see
  References") don't reliably scroll to target, a small but constant paper cut.
- **Stale `ipywidgets` support** — interactive sliders (`@interact`) are core
  to how these tutorials teach, and Colab's widget support lags upstream.

The bar for any replacement is high: **reliably one-click, no-install, and
editable, every time** — "occasionally brilliant, occasionally broken is worse
than Colab as it stands."

## Platforms considered

| Platform | Backing / durability | Live, editable Python in browser? | Cross-notebook navigation | Verdict |
|---|---|---|---|---|
| **marimo** (piloted) | marimo Inc., **acquired by CoreWeave** (Oct 2025); OSS reportedly stays free, team retained. Fast release cadence, at least one documented breaking change, several open WASM-export bugs (below). | **Yes** — first-class WASM/Pyodide export, reactive dependency graph, this is marimo's core product. | Not built-in; hand-rolled (this pilot's `docs/index.html` + relative links). | Hands-on verified for T1–T4. Real rough edges, all worked around (see below). Strategic risk from the acquisition and fast-moving OSS is real but not (yet) disqualifying. |
| **JupyterLite** (optionally wrapped in Jupyter Book's nav) | Jupyter-governed (nonprofit-adjacent), not VC-backed — the strongest institutional-durability profile of any option here. | Yes — a real Jupyter Notebook/Lab UI compiled to WASM/Pyodide, the direct Jupyter-ecosystem counterpart to marimo's WASM export. | Not native to JupyterLite itself; **Jupyter Book can supply it** (built-in site/TOC, `{toggle}` admonitions for answer-hiding), but Jupyter Book by itself is a *static* renderer — since nearly every page in this course needs to be live, Book's main contribution here would be the nav/search/TOC chrome around live JupyterLite kernels, not the interactivity itself (see below). | **Not hands-on tested** — only bare scaffolding exists in `pilot/jupyterbook/`. Cited, real bug: `ipywidgets.interact()` breaks after the first slider move inside JupyterLite ([ipywidgets#3935](https://github.com/jupyter-widgets/ipywidgets/issues/3935)) — directly in the widget/kernel bridge this course leans on hardest, a direct parallel to marimo's own rough edges. Jupyter Book is also mid-migration to a new `mystmd` backend (near-term authoring churn). |
| **Quarto** (+ `quarto-live` / `quarto-pyodide`) | Quarto core: Posit PBC (public-benefit corp, MIT-licensed, ~5 FTE on Quarto). Strong institutional backing for the *publishing* layer. **But** live in-browser Python execution is not native to Quarto — it comes from third-party community extensions ([`r-wasm/quarto-live`](https://github.com/r-wasm/quarto-live), or the independent [`quarto-pyodide`](https://quarto.thecoatlessprofessor.com/pyodide/)), which carry their own, much smaller-team durability profile, more comparable to a solo-maintained pilot than to Quarto itself. Quarto 2 (a full Rust rewrite) is planned for late 2026 — a concrete near-term churn risk, same category as Jupyter Book's `mystmd` move. | Only via the above extensions; not core Quarto. Two *competing*, independent extensions exist for the same job — a fragmentation signal, not consolidation. | **Best native story of any option** — Quarto websites/books generate sidebar nav, next/prev links, and a real TOC automatically; this pilot had to hand-roll exactly that for marimo (`docs/index.html`, manual relative "Next" links). | **Not hands-on tested.** On paper, most attractive for authoring/navigation/multi-format output (HTML, PDF, slides); most uncertain on the one capability that matters most here (live, editable, reactive Python), since that's bolted on by a third party. Worth a real pilot before ruling out. |
| **Binder** | Uncertain long-term funding (Simula's listed support window reads as "2015–2025", i.e. possibly lapsed; no confirmed 2024–2026 funding found either way) but operating as of Aug 2026. | Yes — a real Jupyter+MathJax server, so it sidesteps *all* the WASM-specific rendering/dependency quirks below by construction. | None built-in beyond a normal Jupyter file browser. | Solves the hardest problems (macros, `ipywidgets`, dependency drift) simply by not running in a browser sandbox at all — the tradeoff is a real server: 5–10 min cold container build for a numpy/scipy-scale stack, longer with DAPPER. Not hands-on tested for this repo's actual dependency set. |
| **2i2c JupyterHub** | Credible nonprofit operator. | Yes (hosted JupyterHub). | Depends on setup. | ~$1500/month cited cost + a subsidy-application process — likely disqualifying for a solo-maintained course, on cost/overhead grounds alone. Not investigated further. |
| **GitHub Codespaces** | Strong institutional backing (Microsoft/GitHub). | Yes (full VM, not sandboxed WASM). | None built-in. | Free tier: 120 core-hours/month (60 hrs on a 2-core box) for public repos, with some community confusion about the real usable hours. No hard cold-start number found; default image preinstalls numpy/scipy/pandas/matplotlib. No strong evidence a GitHub login is a worse barrier than a Google one for students. Not hands-on tested. |

**"Jupyter Book" vs. "JupyterLite" — these are two different things.** Jupyter
Book alone renders notebooks to a static site at build time — code and
outputs baked in, read-only. It does not, by itself, satisfy "students can
edit and run code." The actual interactivity comes from **JupyterLite** (a
real Jupyter Notebook/Lab UI compiled to WASM), which Jupyter Book can embed
live inside its pages via `jupyterlite-sphinx`. So it's not that "the book
becomes a plain notebook" — you'd get a book-style nav/TOC/search shell
wrapped around live JupyterLite kernels, which is a real structural advantage
(it's exactly the cross-notebook navigation this pilot had to hand-roll for
marimo). But since nearly every page here needs to be live rather than mostly
static prose, Jupyter Book's headline strength (fast static rendering) barely
applies — what actually matters is whether **JupyterLite** holds up, which is
why the table above treats it as the primary row.

**Honest caveat**: only marimo has had the full hands-on treatment (clean-room
rebuild, browser testing across cold loads, actual defect-finding). Every
other row is a paper evaluation. Before this becomes a final decision, the
strongest alternatives (Jupyter Book+JupyterLite for durability, Quarto for
authoring/navigation *if* `quarto-live`/`quarto-pyodide` proves solid) deserve
the same hands-on scrutiny marimo got.

## What was built

A pilot conversion of T1–T4 to marimo notebooks, hosted as a public,
self-contained, editable WASM export on GitHub Pages
(`patnr/da-tuts-marimo-pilot`, `docs/T1/T1.html` … `docs/T4/T4.html`, linked
from a hand-built `docs/index.html`). Every notebook opens with no install, no
login, no server — sliders, plots, and code are all editable and re-runnable
directly in the browser. `show_answer(tag)` and `hookup(controls, plot_fn)`
helpers (in `answers_data.py` / `notebook_utils.py`) collapse what would
otherwise be repetitive per-exercise boilerplate into one-line calls.

## Challenges overcome or worked around (marimo pilot)

Each of these was found and fixed hands-on, not anticipated in advance:

- **LaTeX macro preamble is fundamentally incompatible with marimo's rendering
  model** — each `$...$` fragment renders through an independent KaTeX call
  with no shared macro namespace, so the original notebooks' `\newcommand`
  shorthand (`\x`, `\Expect`, etc.) showed up as literal red error text
  everywhere except the one cell that happened to define it. **Resolution**:
  dropped the macro system entirely and hand-expanded every shorthand to
  literal KaTeX (`\mathscr{N}`, `\mathbb{E}`, `\mathbf{x}`, …) across T1–T4 and
  `answers_data.py`. More authoring effort per notebook, but zero rendering
  risk.
- **Raw `<details>`/`<summary>` HTML silently suppresses KaTeX rendering** for
  any math inside it — a general marimo limitation, confirmed via isolated
  repro, unrelated to the macro issue. **Resolution**: every "optional
  reading" aside containing math was converted to `mo.accordion({title:
  mo.md(body)})`, which renders correctly in both title and body.
- **Math must start its own line to render** — `$$...$$` glued onto the end of
  a sentence, or nested inside a list item's continuation line, silently fails
  to render. **Resolution**: single-line math stays inline (`$...$`); display
  equations were pulled onto their own paragraph with `$$` alone on open and
  close lines, across T1–T4 and `answers_data.py`.
- **`plt.ion()` as a cell's last statement auto-displays its `ExitStack`
  repr** — marimo's "last expression is the output" rule fires before the
  auto-generated `return`. **Resolution**: `_ = plt.ion()`.
- **`mo.ui.batch()` crashes on LaTeX braces** — it templates markdown via
  literal `str.format()`, so any `\mathbf{x}`-style brace in the surrounding
  prose collides with the templating syntax (`KeyError`). **Resolution**:
  reverted to `mo.ui.dictionary` + `mo.hstack`/`mo.vstack` (via the `hookup()`
  helper) instead of the more Jupyter-`@interact`-like inline templating.
- **`dapper==1.7.3` fails to install under Pyodide/WASM** — it requires
  `dill>=0.4.1`, Pyodide ships `dill==0.3.8` preinstalled, and micropip's
  default install mode won't upgrade an already-installed package, aborting
  the *entire* install batch for that cell (25 cascaded errors in T1 alone).
  **Resolution (this pilot)**: removed the DAPPER-dependent section from T1
  with an explanatory migration note; not a real fix (see below).
- **Exporting multiple notebooks with shared local-module imports into the
  same output directory silently collides** — marimo's WASM export
  auto-builds a wheel per local sibling-module import (including
  notebook-to-notebook imports via `App.embed()`) into
  `<output>/public/wheels/`, but each export run overwrites that folder based
  only on *its own* needs, deleting wheels a sibling notebook still needs.
  **Resolution**: export every notebook into its own isolated subdirectory
  (`docs/T1/`, `docs/T2/`, …) so each gets its own `public/wheels/` — sidesteps
  the bug rather than fixing it.
- **GitHub Pages (Jekyll) silently drops every underscore-prefixed file** —
  including assets marimo's own export generates — causing a blank white page
  with no error. marimo's per-notebook `.nojekyll` isn't at the Pages *source
  root*, so it doesn't help. **Resolution**: a `.nojekyll` at `docs/` itself.
- **`--mode run` (clean, hidden code) vs. `--mode edit` (editable, but
  `hide_code=True` cells render as translucent read-only previews instead of
  being invisible)** — a real trade-off, not a bug to fix. Editability is the
  whole point of this migration, so `--mode edit` was chosen and the
  limitation ([marimo#5244](https://github.com/marimo-team/marimo/issues/5244),
  open upstream) was disclosed rather than chased. Adding one-line
  `show_answer()`/`hookup()` helpers incidentally makes the translucent
  preview much less obtrusive, since there's a lot less code to see through.
- **Cross-notebook import via an extracted module broke pedagogical
  continuity** — students had already seen T2's actual code, not a
  `shared_gaussian.py` stand-in. **Resolution**: marimo's `App.embed()` API
  (`from T2 import app as _t2_app; await _t2_app.embed()`) runs T2 itself and
  exposes its variables via `.defs`, so T3 imports the real notebook. Surfaced
  a pre-existing authoring bug in the process (`pdf_U1`, `mean_and_var` were
  defined in T2 but never `return`ed from their cells).
- **Manual `<a name=...>` anchors (a Colab-era workaround) turned out to still
  matter, just differently than assumed** — marimo *does* auto-generate
  heading ids, but as a lowercase-hyphenated slug, not matching the manual
  anchors' exact-case names. Removing the anchors without updating the links
  that depended on them would have silently broken in-page navigation.
  **Resolution**: removed all manual anchors (T1: 1, T2: 6) and fixed the
  handful of dependent links to the real generated ids — verified by clicking
  them in a live rendered export.
- **No built-in multi-notebook navigation** — solved for this pilot with a
  hand-built `docs/index.html` landing page and real relative "Next: T…"
  links between notebooks (`../T2/T2.html`, etc.) — verified to navigate
  correctly across the `docs/T*/T*.html` layout on GitHub Pages.

## Challenges anticipated ahead

- **T5–T9 are not yet converted.** 99 more answer entries to port, and 17
  `interact()` call sites across T2, T3, T4×2, T5×2, T6×5, T7×4, T8×2 using a
  nested-layout DSL (`top=`/`right=`/`bottom=`/`left=` with nested lists) that
  marimo has no native equivalent for. `hookup()` now supports the flat
  (non-nested) version of that same idea, but the more elaborate T5-style
  layouts haven't been hands-on translated yet.
- **The DAPPER/`dill` WASM conflict is unresolved**, not just worked around.
  Since the user maintains DAPPER itself, relaxing/testing the `dill>=0.4.1`
  pin against Pyodide's `0.3.8` is a cheap, concrete next step — but until
  it's done, T1's WASM export ships without its DAPPER/EnKF section, and T6/T9
  (also DAPPER-dependent) inherit the same blocker.
- **A separate, unresolved DAPPER/matplotlib collision** — even outside WASM,
  in a normal local venv, DAPPER's `xp.stats.replay()` liveplotting collides
  with marimo's custom matplotlib capture backend
  (`MarimoExceptionRaisedError: Figure 1 already exists`); `plt.close('all')`
  did not fix it.
- **The wheel-collision workaround is a manual convention** (one subdirectory
  per notebook), not a real fix — it must be remembered and applied
  consistently as more notebooks are added; nothing currently enforces it.
- **The translucent-hidden-code limitation in `--mode edit`
  ([marimo#5244](https://github.com/marimo-team/marimo/issues/5244)) is open
  upstream** with no available fix short of dropping `hide_code` entirely
  (rejected so far, since it would expose all the accordion/widget-setup
  boilerplate this pilot worked to hide).
- **marimo's pace and ownership are real, ongoing risks**: a fast release
  cadence with at least one documented breaking change (`mo.stop`, 0.20.0),
  several open WASM-export bugs found in the wild (service-worker race
  [#5304](https://github.com/marimo-team/marimo/issues/5304), export 404s
  after a version bump [#6343](https://github.com/marimo-team/marimo/issues/6343),
  files missing offline on iOS [#5206](https://github.com/marimo-team/marimo/issues/5206),
  an Altair WASM-init TypeError [#9152](https://github.com/marimo-team/marimo/issues/9152)),
  and the October 2025 CoreWeave acquisition, which makes marimo's
  teaching-notebook export a minor feature of a leveraged GPU-cloud company
  rather than that company's core product.
- **No CI for the docs/ rebuild.** Every source change currently requires
  manually re-running `marimo export html-wasm` per notebook and re-pushing
  `docs/` by hand; nothing catches a forgotten rebuild before it reaches
  GitHub Pages.

## Bottom line

marimo is proven, hands-on, for T1–T4: every defect found so far has a
concrete fix or disclosed trade-off, none are fundamental blockers, and the
result genuinely meets the "one-click, editable, no-install" bar for those
four notebooks. It is not yet a clean, final win: the DAPPER/WASM conflict is
unresolved, T5–T9 are unconverted, and the strongest alternatives (Jupyter
Book+JupyterLite, Quarto) haven't had equivalent hands-on scrutiny — only a
paper evaluation. The most valuable next step, before treating marimo as the
final answer, is giving at least one of those alternatives the same
clean-room, defect-hunting treatment this pilot gave marimo.
