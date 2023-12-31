% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/internal.plasso.R
\name{handle_weights}
\alias{handle_weights}
\title{Sanitizes potential sample weights}
\usage{
handle_weights(w, n)
}
\arguments{
\item{w}{Vector or n x 1 matrix of weights or null if no weights provided}

\item{n}{Number of observations}
}
\value{
Vector of weights.
}
\description{
\code{\link{handle_weights}} cleans potential sample weights or codes them as ones if they are not specified.
}
\keyword{internal}
