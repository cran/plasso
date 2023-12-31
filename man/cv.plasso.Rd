% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv.plasso.R
\name{cv.plasso}
\alias{cv.plasso}
\title{Cross-Validated Lasso and Post-Lasso}
\usage{
cv.plasso(x, y, w = NULL, kf = 10, parallel = FALSE, ...)
}
\arguments{
\item{x}{Matrix of covariates (number of observations times number of covariates matrix)}

\item{y}{Vector of outcomes}

\item{w}{Vector of weights}

\item{kf}{Number of folds in k-fold cross-validation}

\item{parallel}{Set as TRUE for parallelized cross-validation. Default is FALSE.}

\item{...}{Pass \code{\link[glmnet]{glmnet}} options}
}
\value{
cv.plasso object (using a list structure) including the base \code{\link[glmnet]{glmnet}} object and cross-validation results (incl. optimal Lambda values) for both Lasso and Post-Lasso model.
\item{call}{the call that produced this}
\item{lasso_full}{base \code{\link[glmnet]{glmnet}} object}
\item{kf}{number of folds in k-fold cross-validation}
\item{cv_MSE_lasso}{cross-validated MSEs of Lasso model (for every iteration of k-fold cross-validation)}
\item{cv_MSE_plasso}{cross-validated MSEs of Post-Lasso model (for every iteration of k-fold cross-validation)}
\item{mean_MSE_lasso}{averaged cross-validated MSEs of Lasso model}
\item{mean_MSE_plasso}{averaged cross-validated MSEs of Post-Lasso model}
\item{ind_min_l}{index of MSE optimal lambda value for Lasso model}
\item{ind_min_pl}{index of MSE optimal lambda value for Post-Lasso model}
\item{lambda_min_l}{MSE optimal lambda value for Lasso model}
\item{lambda_min_pl}{MSE optimal lambda value for Post-Lasso model}
\item{names_l}{Names of active variables for MSE optimal Lasso model}
\item{names_pl}{Names of active variables for MSE optimal Post-Lasso model}
\item{coef_min_l}{Coefficients for MSE optimal Lasso model}
\item{coef_min_pl}{Coefficients for MSE optimal Post-Lasso model}
\item{x}{Input matrix of covariates}
\item{y}{Matrix of outcomes}
\item{w}{Matrix of weights}
}
\description{
\code{\link{cv.plasso}} uses the \code{\link[glmnet]{glmnet}} package to estimate the coefficient paths and cross-validates least squares Lasso AND Post-Lasso.
}
\examples{
# load toeplitz data
data(toeplitz)
# extract target and features from data
y = as.matrix(toeplitz[,1])
X = toeplitz[,-1]
# fit cv.plasso to the data
\donttest{p.cv = plasso::cv.plasso(X,y)}
# get basic summary statistics
\donttest{print(summary(p.cv, default=FALSE))}
# plot cross-validated MSE curves and number of active coefficients
\donttest{plot(p.cv, legend_pos="bottomleft")}
# get coefficients at MSE optimal lambda value for both Lasso and Post-Lasso model
\donttest{coef(p.cv)}
# get coefficients at MSE optimal lambda value according to 1-standard-error rule
\donttest{coef(p.cv, se_rule=-1)}
# predict fitted values along whole lambda sequence 
\donttest{pred = predict(p.cv, s="all")}
\donttest{head(pred$plasso)}

}
