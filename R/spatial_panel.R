# ──────────────────────────────────────────────
# Spatial Panel Model (SAR / SEM) via splm
# Called by: src/ngforecast/models/fundamentals/fit.py
# ──────────────────────────────────────────────
# Usage: Rscript R/spatial_panel.R <panel.csv> <W.csv> <output.json> <model> <estimator> <effects> <covariates>

suppressPackageStartupMessages({
  library(splm)
  library(spdep)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

panel_path  <- args[1]
w_path      <- args[2]
output_path <- args[3]
model_type  <- ifelse(length(args) >= 4, args[4], "SAR")
estimator   <- ifelse(length(args) >= 5, args[5], "ML")
effects     <- ifelse(length(args) >= 6, args[6], "twoways")
covariates  <- ifelse(length(args) >= 7, strsplit(args[7], ",")[[1]], c())

cat("Loading data...\n")
panel <- read.csv(panel_path, stringsAsFactors = FALSE)
W_mat <- as.matrix(read.csv(w_path, header = FALSE))

# Build spatial weights
n <- nrow(W_mat)
listw <- mat2listw(W_mat, style = "W")

# Build formula
if (length(covariates) == 0) {
  covariates <- c("vote_share_lag1", "poverty_rate", "unemployment_rate")
}
formula_str <- paste("vote_share ~", paste(covariates, collapse = " + "))
cat("Formula:", formula_str, "\n")

# Convert to pdata.frame
panel$state <- as.factor(panel$state)
panel$year  <- as.factor(panel$year)

# Remove rows with NAs in covariates
complete_vars <- c("vote_share", "state", "year", covariates)
panel_clean <- panel[complete.cases(panel[, complete_vars, drop = FALSE]), ]

pdata <- pdata.frame(panel_clean, index = c("state", "year"))

cat("Fitting", model_type, "model with", estimator, "estimator...\n")

tryCatch({
  if (model_type == "SAR") {
    fit <- spml(
      formula = as.formula(formula_str),
      data    = pdata,
      listw   = listw,
      model   = "within",
      spatial.error = FALSE,
      lag     = TRUE,
      effect  = effects
    )
  } else if (model_type == "SEM") {
    fit <- spml(
      formula = as.formula(formula_str),
      data    = pdata,
      listw   = listw,
      model   = "within",
      spatial.error = TRUE,
      lag     = FALSE,
      effect  = effects
    )
  } else {
    stop(paste("Unknown model type:", model_type))
  }

  cat("Model fitted successfully.\n")
  cat(summary(fit))

  # Extract results
  coefs <- as.list(coef(fit))
  results <- list(
    model       = model_type,
    estimator   = estimator,
    effects     = effects,
    n_obs       = nrow(panel_clean),
    coefficients = coefs
  )

  # Add spatial parameter if available
  if (!is.null(fit$arcoef)) {
    results$rho <- fit$arcoef
  }

  write(toJSON(results, auto_unbox = TRUE, pretty = TRUE), output_path)
  cat("Results written to", output_path, "\n")

}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  error_result <- list(error = conditionMessage(e))
  write(toJSON(error_result, auto_unbox = TRUE), output_path)
  quit(status = 1)
})
