# ──────────────────────────────────────────────
# GMM Dynamic Panel (Arellano–Bond / System-GMM)
# Called by: src/ngforecast/models/gmm/fit.py
# ──────────────────────────────────────────────
# Usage: Rscript R/gmm.R <panel.csv> <output.json> <model_type> <lags>

suppressPackageStartupMessages({
  library(plm)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

panel_path  <- args[1]
output_path <- args[2]
model_type  <- ifelse(length(args) >= 3, args[3], "system_gmm")
n_lags      <- ifelse(length(args) >= 4, as.integer(args[4]), 2)

cat("Loading data...\n")
panel <- read.csv(panel_path, stringsAsFactors = FALSE)
panel$state <- as.factor(panel$state)
panel$year  <- as.integer(panel$year)

# Build pdata.frame
pdata <- pdata.frame(panel, index = c("state", "year"))

formula_str <- "vote_share ~ lag(vote_share, 1) + poverty_rate + unemployment_rate"
cat("Formula:", formula_str, "\n")

tryCatch({
  if (model_type == "system_gmm") {
    cat("Fitting System-GMM...\n")
    fit <- pgmm(
      as.formula(paste(formula_str, "| lag(vote_share, 2:99)")),
      data        = pdata,
      effect      = "twoways",
      model       = "twosteps",
      transformation = "ld"  # system GMM
    )
  } else {
    cat("Fitting Difference-GMM...\n")
    fit <- pgmm(
      as.formula(paste(formula_str, "| lag(vote_share, 2:99)")),
      data        = pdata,
      effect      = "twoways",
      model       = "twosteps",
      transformation = "d"
    )
  }

  cat("Model fitted successfully.\n")
  cat(summary(fit, robust = TRUE))

  coefs <- as.list(coef(fit))
  results <- list(
    model        = model_type,
    n_lags       = n_lags,
    n_obs        = nrow(panel),
    coefficients = coefs
  )

  # Sargan/Hansen test
  sargan <- tryCatch(sargan(fit), error = function(e) NULL)
  if (!is.null(sargan)) {
    results$sargan_p <- sargan$p.value
  }

  write(toJSON(results, auto_unbox = TRUE, pretty = TRUE), output_path)
  cat("Results written to", output_path, "\n")

}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  error_result <- list(error = conditionMessage(e))
  write(toJSON(error_result, auto_unbox = TRUE), output_path)
  quit(status = 1)
})
