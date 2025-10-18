# Not used. 

import numpy as np, emcee, scipy.stats as st

def log_prior(theta):
    m, b, log_sigma, logit_p = theta
    if -1e3 < m < 1e3 and -1e6 < b < 1e6 and -10 < log_sigma < 3:
        # weak priors: N(0,1) on log sigma  and on logit(p)
        return -0.5*(log_sigma**2 + logit_p**2)
    return -np.inf

def log_likelihood(theta, x, y, rho_b):
    m, b, log_sigma, logit_p = theta
    sigma = np.exp(log_sigma)
    p     = 1/(1+np.exp(-logit_p))        # logistic → (0,1)

    d_perp = (m*x - y + b)/np.hypot(m,1.0)
    gauss  = st.norm.logpdf(d_perp, 0.0, sigma)

    # log-sum-exp for mixture:  log(p*G + (1-p)*rho_b)
    a = np.logaddexp(np.log(p) + gauss,
                     np.log1p(-p) + np.log(rho_b))
    return a.sum()

def log_prob(theta, x, y, rho_b):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, rho_b)

def fit_rc_mixture(x, y, rho_b=1e-4,
                   nwalkers=40, nsteps=6000, burnin=1000):
    # rough initial guesses
    m0, b0 = np.polyfit(x, y, 1)
    theta0 = np.array([m0, b0, np.log(np.std(y)*0.3), 0.0])
    pos    = theta0 + 1e-4*np.random.randn(nwalkers, 4)

    sampler = emcee.EnsembleSampler(
        nwalkers, 4, log_prob, args=(x, y, rho_b)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=burnin, flat=True)
    m_med, b_med = np.median(samples[:,0]), np.median(samples[:,1])
    m_err        = np.std   (samples[:,0], ddof=1)

    return dict(slope=m_med,
                slope_err=m_err,
                intercept=b_med,
                samples=samples)

