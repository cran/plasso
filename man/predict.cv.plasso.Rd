% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv.plasso.R
\name{predict.cv.plasso}
\alias{predict.cv.plasso}
\title{Predict after cross-validated (Post-) Lasso}
\usage{
\method{predict}{cv.plasso}(
  object,
  ...,
  newx = NULL,
  type = c("response", "coefficients"),
  s = c("optimal", "all"),
  se_rule = 0
)
}
\arguments{
\item{object}{Fitted \code{\link{cv.plasso}} model object}

\item{...}{Pass generic \code{\link[stats]{predict}} options}

\item{newx}{Matrix of new values for x at which predictions are to be made. If no value is supplied, x from fitting procedure is used. This argument is not used for \code{type="coefficients"}.}

\item{type}{Type of prediction required. \code{"response"} returns fitted values, \code{"coefficients"} returns beta estimates.}

\item{s}{Determines whether prediction is done for all values of lambda (\code{"all"}) or only for the optimal lambda (\code{"optimal"}) according to the standard error-rule.}

\item{se_rule}{If equal to 0, predictions from cross-validated MSE minimum (default). Negative values go in the direction of smaller
models, positive values go in the direction of larger models (e.g. \code{se_rule=-1} creates the standard 1SE rule).
This argument is not used for \code{s="all"}.}
}
\value{
List object containing either fitted values or coefficients for both
the Lasso and Post-Lasso models respectively.
\item{lasso}{Matrix with Lasso predictions or coefficients}
\item{plasso}{Matrix with Post-Lasso predictions or coefficients}
}
\description{
Prediction for cross-validated (Post-) Lasso.
}
\examples{

# load toeplitz data
data(toeplitz)
# extract target and features from data
y = as.matrix(toeplitz[,1])
X = toeplitz[,-1]
# fit cv.plasso to the data
\donttest{p.cv = plasso::cv.plasso(X,y)}
# predict fitted values along whole lambda sequence 
\donttest{pred = predict(p.cv, s="all")}
\donttest{head(pred$plasso)}
# predict fitted values for optimal lambda value (according to cross-validation) 
\donttest{pred_optimal = predict(p.cv, s="optimal")}
\donttest{head(pred_optimal$plasso)}
# predict fitted values for new feature set X
\donttest{X_new = head(X, 10)}
\donttest{pred_new = predict(p.cv, newx=X_new, s="optimal")}
\donttest{pred_new$plasso}
# get estimated coefficients along whole lambda sequence
\donttest{coefs = predict(p.cv, type="coefficients", s="all")}
\donttest{head(coefs$plasso)}
# get estimated coefficients for optimal lambda value according to 1-standard-error rule
\donttest{predict(p.cv, type="coefficients", s="optimal", se_rule=-1)}

}
