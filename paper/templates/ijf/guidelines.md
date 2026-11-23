# International Journal of Forecasting — Paper Writing Guidelines

Target: a full research article for the *International Journal of Forecasting*
(Elsevier). Verify current limits against the IJF Guide for Authors before
submission; the rules below govern this draft.

## General Setup
* **Document class:** `\documentclass[preprint,12pt,authoryear]{elsarticle}` (already in the template; do not change).
* Packages already loaded: amsmath, amssymb, graphicx, booktabs, hyperref, url. Do not add packages that are not in a standard TeX distribution.
* Title, authors and affiliations go in the `frontmatter`. Keep the author placeholder exactly as in the template: "Anonymous Author", "Institution".

## Length and Structure
* Aim for roughly 8,000–10,000 words of main text (excluding references and appendix).
* **Abstract:** one paragraph, at most about 200 words, no citations.
* **Keywords:** 4–6 keywords separated by `\sep`.
* Keep the section order of the template. Subsections are encouraged.

## Citations and References
* Author–year citations via natbib (loaded by elsarticle with `authoryear`):
  `\citet{key}` → "Author (2020)", `\citep{key}` → "(Author, 2020)".
* Bibliography: `\bibliographystyle{elsarticle-harv}` and `\bibliography{references}` (already in the template).
* Cite only entries that exist in the provided citation library.

## Figures and Tables
* All figures are in the `figures/` directory. Always use the prefix, e.g. `\includegraphics[width=\linewidth]{figures/fig1_backtest_crps.png}`.
* Figure captions go **below** figures; table captions go **above** tables. Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
* Each figure and table must be referenced in the text.

## Scientific Integrity Rules (critical — override any other instinct)
1. **Every number in the paper must come verbatim from the experimental log.** Do not invent, round differently in a misleading way, extrapolate, or "improve" any result. If a number is not in the log, do not state it.
2. **Report negative and mixed results as stated in the log.** In particular: the spatial (SAR) term did not improve out-of-sample forecasts; uniform swing is a strong point-forecast baseline; the national swing dominates the error and the national forecast is weak; P(§134 spread met) was under-forecast in the backtests.
3. **Disclose the model-development history exactly as in the log**: the level model (v1a) was replaced by swing models (v1b) after seeing backtest results; the v2 protocol was frozen before the v2 run and uses prequential stacking whose weights never see the target election; one post-freeze computational change (sampler target_accept) is logged.
4. Do not claim the model predicts who will win in 2027 or the runoff probability — it does not. It forecasts the incumbent party's (APC) vote shares and the §134 spread requirement.
5. Use cautious, precise language for causal or substantive claims about Nigerian politics; the paper is about forecasting methodology and evaluation.
6. Distinguish "forecast mode" (model's own national forecast) from "oracle mode" (true national share supplied) every time results are reported.

## Math Notation
* Use standard notation: $y_{it}$ for the incumbent-party share in state $i$ at election $t$, $N_t$ for the national share, $d_{it}=\operatorname{logit} y_{it}-\operatorname{logit} N_t$ for the state deviation, $W$ for the spatial weight matrix, $\rho$ for the spatial parameter.

## Optional Statements (end of main text, before references)
* **Data and code availability:** state that code, curated data, the frozen evaluation protocol and the pre-registered forecast are available in a public repository (do not give a URL; write "[repository URL withheld for review]").
* **Declaration of competing interest:** "The author declares no competing interests."
