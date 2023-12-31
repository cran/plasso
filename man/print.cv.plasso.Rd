% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv.plasso.R
\name{print.cv.plasso}
\alias{print.cv.plasso}
\title{Print cross-validated (Post-) Lasso model}
\usage{
\method{print}{cv.plasso}(x, ..., digits = max(3, getOption("digits") - 3))
}
\arguments{
\item{x}{\code{\link{cv.plasso}} object}

\item{...}{Pass generic \code{\link[base]{print}} options}

\item{digits}{Integer, used for number formatting}
}
\value{
Prints basic statistics for different lambda values of a fitted \code{\link{plasso}} object,
i.e. cross-validated MSEs for both Lasso and Post-Lasso model as well as the number of active variables.
}
\description{
Printing main insights from cross-validated (Post-) Lasso model.
}
