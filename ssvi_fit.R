require(rugarch)
require(rmgarch)
require(xts)
require(urca)
require(vars)

rm(list=ls()) # clear all variables

r2_score <- function(actual, pred) {
  num = sum((actual - pred)^2)
  den = sum((actual - mean(actual))^2)
  return(1 - num/den)
}

metrics <- function(alpha.actual, alpha.predicted, beta.actual, beta.predicted, mu.actual, mu.predicted, rho.actual, rho.predicted, nu.actual, nu.predicted) {
    # Compute the RMSE for each variable
    alpha.rmse <- sqrt(sum((as.numeric(alpha.actual) - alpha.predicted)**2)/length(alpha.actual))
    print(paste("RMSE for alpha is ", alpha.rmse))
    print(paste("R2 for alpha is ", r2_score(as.numeric(alpha.actual), alpha.predicted)))
    beta.rmse <- sqrt(sum((as.numeric(beta.actual) - beta.predicted)**2)/length(beta.actual))
    print(paste("RMSE for beta is ", beta.rmse))
    print(paste("R2 for beta is ", r2_score(as.numeric(beta.actual), beta.predicted)))
    mu.rmse <- sqrt(sum((as.numeric(mu.actual) - mu.predicted)**2)/length(mu.actual))
    print(paste("RMSE for mu is ", mu.rmse))
    print(paste("R2 for mu is ", r2_score(as.numeric(mu.actual), mu.predicted)))
    rho.rmse <- sqrt(sum((as.numeric(rho.actual) - rho.predicted)**2)/length(rho.actual))
    print(paste("RMSE for rho is ", rho.rmse))
    print(paste("R2 for rho is ", r2_score(as.numeric(rho.actual), rho.predicted)))
    nu.rmse <- sqrt(sum((as.numeric(nu.actual) - nu.predicted)**2)/length(nu.actual))
    print(paste("RMSE for nu is ", nu.rmse))
    print(paste("R2 for nu is ", r2_score(as.numeric(nu.actual), nu.predicted)))
}

arifma_garch <- function() {
  # Load the data
  ssvi_params_SPX_call = read.csv("ssvi_params_SPX_call.csv")
  
  u <- xts(ssvi_params_SPX_call, order.by = as.POSIXct(ssvi_params_SPX_call$date))
  
  # Build the spec for fitting the autoregressive systems
  spec = alpha.spec = ugarchspec(mean.model = 
                                   list(armaOrder = c(1,1), include.mean=FALSE), 
                                 distribution.model = "std")
  
  # Total size used to test
  total.size = length(u$alpha)
  # How many to keep for out of sample testing
  out.sample = total.size - 3000 # 3000 samples for training rest fo testing
  
  # Make the fit for alpha, beta, mu, rho using the above spec for autoregressive systems
  alpha.fit = ugarchfit(spec, data = u$alpha[1:total.size], out.sample = out.sample)
  beta.fit = ugarchfit(spec, data = u$beta[1:total.size], out.sample = out.sample)
  mu.fit = ugarchfit(spec, data = u$mu[1:total.size], out.sample = out.sample)
  
  rho.spec = ugarchspec(mean.model = list(armaOrder = c(1,1), include.mean=FALSE), 
                       distribution.model = "std", variance.model = list(garchOrder = c(1,1)))
  rho.fit = ugarchfit(rho.spec, data = u$rho[1:total.size], out.sample = out.sample)
  
  # This is the spec for nu (because it is special)
  nu.spec = ugarchspec(mean.model = list(armaOrder = c(2,2), include.mean=FALSE), 
                       distribution.model = "std", variance.model = list(garchOrder = c(1,1)))
  nu.fit = ugarchfit(nu.spec, data = u$nu[1:total.size], out.sample = out.sample)
  
  # Show the results of the fit for checking that they all look good.
  alpha.fit
  beta.fit
  mu.fit
  rho.fit
  nu.fit
  
  # Now start forecasting for each variable
  ahead = 1
  alpha.forecast = ugarchforecast(alpha.fit, n.ahead = ahead, n.roll = out.sample-ahead)
  beta.forecast = ugarchforecast(beta.fit, n.ahead = ahead, n.roll = out.sample-ahead)
  mu.forecast = ugarchforecast(mu.fit, n.ahead = ahead, n.roll = out.sample-ahead)
  rho.forecast = ugarchforecast(rho.fit, n.ahead = ahead, n.roll = out.sample-ahead)
  nu.forecast = ugarchforecast(nu.fit, n.ahead = ahead, n.roll = out.sample-ahead)
  
  # Compare the outputs using different metrics
  alpha.actual = u$alpha[(total.size-out.sample+1):(total.size)]
  alpha.predicted = t(alpha.forecast@forecast$seriesFor)
  beta.actual = u$beta[(total.size-out.sample+1):(total.size)]
  beta.predicted = t(beta.forecast@forecast$seriesFor)
  mu.actual = u$mu[(total.size-out.sample+1):(total.size)]
  mu.predicted = t(mu.forecast@forecast$seriesFor)
  rho.actual = u$rho[(total.size-out.sample+1):(total.size)]
  rho.predicted = t(rho.forecast@forecast$seriesFor)
  nu.actual = u$nu[(total.size-out.sample+1):(total.size)]
  nu.predicted = t(nu.forecast@forecast$seriesFor)
  
  metrics(alpha.actual, alpha.predicted, beta.actual, beta.predicted, 
          mu.actual, mu.predicted, rho.actual, rho.predicted, nu.actual, nu.predicted)
  df = (data.frame("alpha" = alpha.predicted, "beta" = beta.predicted, 
        "mu" = mu.predicted, "rho" = rho.predicted, "nu" = nu.predicted))
  colnames(df) = c("alpha", "beta", "mu", "rho", "nu")
  return(df)
}

# Now we predict using mutivariate VAR + DCC
var_dcc <- function(PLAGS=5, ROBUST=FALSE){
  # Load the data
  ssvi_params_SPX_call = read.csv("ssvi_params_SPX_call.csv")
  # There has to be a better way of doing this.
  u <- xts(ssvi_params_SPX_call, order.by = as.POSIXct(ssvi_params_SPX_call$date))
  u <- u[,c("alpha", "beta", "mu", "rho", "nu")]
  u <- as.data.frame(coredata(u))
  for (i in unique(colnames(u))) {
    u[,i] <- as.numeric(u[,i])
  }
  rownames(u) <- ssvi_params_SPX_call$date
  # Total size used to test
  total.size = length(u$alpha)
  # How many to keep for out of sample testing
  out.sample = total.size - 3000 # 3000 samples for training rest for testing 
  
  # Build the spec for fitting the autoregressive systems
  spec = ugarchspec(mean.model = # This is used for alpha, beta, mu
                      list(armaOrder = c(1,1), include.mean=FALSE), 
                    distribution.model = "std")
  rho.spec = ugarchspec(mean.model = list(armaOrder = c(1,1), include.mean=FALSE), 
                        distribution.model = "std", variance.model = list(garchOrder = c(1,1))) 
  nu.spec = ugarchspec(mean.model = list(armaOrder = c(2,2), include.mean=FALSE), 
                       distribution.model = "std", variance.model = list(garchOrder = c(1,1)))

  # Fit a var object outside
  var_fit <- varxfit(u[1:(total.size-out.sample),], p = PLAGS, postpad = "constant",
                     constant = FALSE, robust = ROBUST)
  ahead = 1
  var_mu_forecast <- varxforecast(u[1:total.size,], var_fit$Bcoef, p = PLAGS, out.sample = out.sample,
                                  n.ahead = ahead, n.roll = out.sample-ahead, mregfor = NULL)
  alpha.actual = u$alpha[(total.size-out.sample+1):(total.size)]
  beta.actual = u$beta[(total.size-out.sample+1):(total.size)]
  mu.actual = u$mu[(total.size-out.sample+1):(total.size)]
  rho.actual = u$rho[(total.size-out.sample+1):(total.size)]
  nu.actual = u$nu[(total.size-out.sample+1):(total.size)]
  
  # Make the multispec
  mspec = rugarch::multispec(c(spec, spec, spec, rho.spec, nu.spec))
  var_dcc <- dccspec(mspec, VAR = TRUE, lag = PLAGS,
                     model = "DCC",
                     lag.criterion = "AIC", dccOrder = c(1,1), distribution = "mvt")
  # var_dcc <- dccspec(mspec, VAR = FALSE, dccOrder = c(1, 1), model = "DCC", distribution = "mvt")
  var_dcc_fit <- dccfit(var_dcc, u[1:total.size,], out.sample = out.sample, solver="nlminb",
                        VAR.fit = var_fit)
  var_dcc_forecast <- dccforecast(var_dcc_fit, n.ahead = ahead, n.roll = out.sample-ahead)
  alpha.predicted.mean = fitted(var_dcc_forecast)[,"alpha",]
  alpha.predicted.var = sigma(var_dcc_forecast)[,"alpha",]
  beta.predicted.mean = fitted(var_dcc_forecast)[,"beta",]
  beta.predicted.var = sigma(var_dcc_forecast)[,"beta",]
  mu.predicted.mean = fitted(var_dcc_forecast)[,"mu",]
  mu.predicted.var = sigma(var_dcc_forecast)[,"mu",]
  rho.predicted.mean = fitted(var_dcc_forecast)[,"rho",]
  rho.predicted.var = sigma(var_dcc_forecast)[,"rho",]
  nu.predicted.mean = fitted(var_dcc_forecast)[,"nu",]
  nu.predicted.var = sigma(var_dcc_forecast)[,"nu",]
  metrics(alpha.actual, alpha.predicted.mean, beta.actual, beta.predicted.mean, 
          mu.actual, mu.predicted.mean, rho.actual, rho.predicted.mean, nu.actual, nu.predicted.mean)
  
  means = t(fitted(var_dcc_forecast)[,,]) # The means
  vars = sqrt(t((sigma(var_dcc_forecast))[,,])) # The std-dev of t-dist

  # The name of the shape parameters
  names = c("[alpha].shape", "[beta].shape", "[mu].shape", "[rho].shape", "[nu].shape")
  shapes = t(as.numeric(coef(var_dcc_fit)[names])) # Get the shape of the t dist
  colnames(shapes) = colnames(means)
  return(list(means = means, vars = vars, shapes = shapes))
}

var_vecm <- function(){
  # Load the data
  ssvi_params_SPX_call = read.csv("ssvi_params_SPX_call.csv")
  # There has to be a better way of doing this.
  u <- xts(ssvi_params_SPX_call, order.by = as.POSIXct(ssvi_params_SPX_call$date))
  u <- u[,c("alpha", "beta", "mu", "rho", "nu")]
  u <- as.data.frame(coredata(u))
  for (i in unique(colnames(u))) {
    u[,i] <- as.numeric(u[,i])
  }
  rownames(u) <- ssvi_params_SPX_call$date
  # Total size used to test
  total.size = length(u$alpha)
  # How many to keep for out of sample testing
  out.sample = total.size - 3000 # 3000 samples for training rest for testing 
  ca.jo.res <- ca.jo(u, type="trace", ecdet="const", K=2, spec="transitory")
  # This confirms that we have complete co-integration, everything is stationary
  print(summary(ca.jo.res))
  
  # Fit a VARS models using the VAR function
  var.model <- VAR(u[1:(total.size-out.sample),], p=2, type="none", lag.max=100, ic="HQ")
  print(summary(var.model))
  normality.test(var.model)
  var.mode.coef <- coef(var.model)
  varcoefs = cbind(var.mode.coef$alpha[, "Estimate"], var.mode.coef$beta[, "Estimate"], var.mode.coef$mu[, "Estimate"], 
                   var.mode.coef$rho[, "Estimate"], var.mode.coef$nu[, "Estimate"])
}

arifmameans = arifma_garch()
varmvs = var_dcc(PLAGS=5)
# Write the outputs
write.csv(arifmameans, "R_arifma_garch_res.csv")
write.csv(varmvs, "R_var_garch_dcc_res.csv")

# VAR_VECM fit using a different package, just to check if everything is OK
# var_vecm()