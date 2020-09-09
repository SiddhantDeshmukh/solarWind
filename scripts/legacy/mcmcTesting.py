import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pymc3 as pm
from pandas_datareader import data
import pandas as pd
import theano.tensor as tt

# plt.style.use('seaborn-darkgrid')
# np.random.seed(123)
#
# # True parameter values
# alpha, sigma = 1, 1
# beta = [1, 2.5]
#
# # Size of dataset
# size = 1000
#
# # Predictor variable
# X1 = np.random.randn(size)
# X2 = np.random.randn(size) * 0.2
#
# # Simulate outcome variable
# Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size) * sigma
#
# # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
# # axes[0].scatter(X1, Y)
# # axes[1].scatter(X2, Y)
# # axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2')
#

# # Initial example
# # Now build the model
# basic_model = pm.Model()
#
# with basic_model:
#     # Priors for unknown parameters
#     alpha = pm.Normal('alpha', mu=0, sd=10)
#     beta = pm.Normal('beta', mu=0, sd=10, shape=2)
#     sigma = pm.HalfNormal('sigma', sd=1)
#
#     # Expected value of outcome
#     mu = alpha + beta[0]*X1 + beta[1]*X2
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
#
#     # Draw samples, can use the 'step' argument to specify sampler (NUTS assigned automatically)
#     step = pm.NUTS()
#     trace = pm.sample(10000, step=step)
#
# # Posterior analysis via posterior plot and summary table
# pm.traceplot(trace)
# print(pm.summary(trace).round(2))
#
# plt.show()

# # Coal mining disasters example
# # Declare data
# disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
#                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
#                             2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
#                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
#                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
#                             3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
#                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
# years = np.arange(1851, 1962)
#
# # plt.plot(years, disaster_data, 'o', markersize=8)
# # plt.ylabel("Disaster count")
# # plt.xlabel("Year")
#
# # Thought to follow a Poisson distribution
# with pm.Model() as disaster_model:
#     switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)
#
#     # Priors for pre- and post-switch rates number of disasters
#     early_rate = pm.Exponential('early-rate', lam=1)
#     late_rate = pm.Exponential('late-rate', lam=1)
#
#     # Allocate appropriate Poisson rates to years before and after current
#     rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
#
#     disasters = pm.Poisson('disasters', mu=rate, observed=disaster_data)
#
#     # Sample data and generate trace; discrete points so can't use NUTS, have to use Metropolis-Hastings
#     step = pm.Metropolis()
#     trace = pm.sample(10000, step=step)
#
# pm.traceplot(trace)
# print(pm.summary(trace))
#
#
# plt.show()

# Try spacecraft data, fit mdot = a(x1)^b first, where x1 is open flux or sn or vr, then try to extend to multiple x's
# Declare files and read in 27-day averaged data as dictionaries
data_dict = pd.read_csv("data/27_day_avg/ALL_SPACECRAFT_27day_avg.csv").to_dict(orient="list")
owens_orig_dict = pd.read_csv('data/models/owens_equatorial.csv').to_dict('list')
omni_dict = pd.read_csv("data/27_day_avg/OMNI_27day_avg_hr.csv").to_dict("list")


ace_mdot = np.array([data for i, data in enumerate(data_dict.get("Mass Loss Rate [g s^-1] (ACE)"))
                     if not np.isnan(data_dict.get("Mass Loss Rate [g s^-1] (ACE)")[i])])
ace_time = np.array([data for i, data in enumerate(data_dict.get("Year (ACE)"))
                     if not np.isnan(data_dict.get("Mass Loss Rate [g s^-1] (ACE)")[i])])
ace_phi = np.array([data for i, data in enumerate(data_dict.get("Open Flux [Mx] (ACE)"))
                     if not np.isnan(data_dict.get("Mass Loss Rate [g s^-1] (ACE)")[i])])
ace_vr = np.array([data for i, data in enumerate(data_dict.get("Radial Wind Velocity [km s^-1] (ACE)"))
                     if not np.isnan(data_dict.get("Mass Loss Rate [g s^-1] (ACE)")[i])]) * 10**5  # Convert to cgs
ace_sn = np.array([data for i, data in enumerate(data_dict.get("Sunspot Number (ACE)"))
                     if not np.isnan(data_dict.get("Mass Loss Rate [g s^-1] (ACE)")[i])])

omni_mdot = np.array([data for i, data in enumerate(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))
                     if not np.isnan(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)")[i])])
omni_time = np.array([data for i, data in enumerate(omni_dict.get("Year (OMNI Hr)"))
                     if not np.isnan(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)")[i])])
omni_phi = np.array([data for i, data in enumerate(omni_dict.get("Open Flux [Mx] (OMNI Hr)"))
                     if not np.isnan(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)")[i])])
omni_vr = np.array([data for i, data in enumerate(omni_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)"))
                     if not np.isnan(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)")[i])]) * 10**5  # Convert to cgs
omni_sn = np.array([data for i, data in enumerate(omni_dict.get("Sunspot Number (OMNI Hr)"))
                     if not np.isnan(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)")[i])])
ace_model = pm.Model()
omni_model = pm.Model()

with omni_model:
    # Independent var(s)
    X1 = ace_phi
    X2 = ace_vr
    # X3 = ace_sn

    # Number of iterations per chain
    niter = 10000

    # Define priors as uniform distributions (no better guess), constraining bounds between operations to improve precision
    a = pm.Uniform('a', lower=1, upper=100)
    b = pm.Uniform('b', lower=0, upper=3)
    c = pm.Uniform('c', lower=1, upper=100)
    d = pm.Uniform('d', lower=0, upper=3)

    # Guess priors to be normally distributed now that I have better guesses; literally using the Pareto outputs
    # a = pm.Normal("a", mu=99.73, sd=3.653)
    # b = pm.Normal("b", mu=0.432, sd=0.001)
    # c = pm.Normal("c", mu=0.549, sd=0.074)

    # Expected value of outcome and standard deviation
    mu = a * (X1**b) + c * (X2**d)
    # mu = np.log10(a) + b*np.log10(X1) + c*np.log10(X2)
    # sigma = pm.HalfNormal('sigma', sd=1)

    # Likelihood (sampling distributions) of observations
    # alpha=1.16 gives the 80-20 rule. look into more alpha values to tailor the power law better
    # Increasing alpha makes tail heavier
    mdot_obs = pm.Pareto('mdot-obs', m=mu, alpha=1.16, observed=ace_mdot)
    # mdot_obs = pm.Lognormal('mdot-obs', mu=mu, tau=1/100, observed=ace_mdot)
    # log_mdot = pm.Normal('log-mdot-obs', mu=mu, sd=sigma, observed=np.log10(ace_mdot))

    # Draw samples
    trace = pm.sample(niter, nchains=2)

# Burn-in period of niter//2 for display purposes
pm.traceplot(trace[niter//2:])
print(pm.summary(trace[niter//2:]))

plt.show()
