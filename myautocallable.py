def payoff(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, RND):

    # INPUT:
    # t_steps : time steps
    # TtM     : time to maturity
    # Drift   : drift list by time
    # Vol     : volatility list by time
    # Disc    : discount rate
    # S_0     : underlying initial value
    # S_k     : kickout barrier
    # S_p     : protection barrier
    # N       : nominal value
    # I       : yearly interest over the nominal
    # RND     : random terms

    # OUTPUT:
    # out     : autocallable structure simulated discounted payoff

    from numpy import exp, sqrt

    # vars
    S_prev = S_0
    TtM[TtM.index[0]-1] = 0 # solve index issue in 'dt'

    # simu
    for t in t_steps:
        # diff = vol[t] * random() # check it # diffusion term

        # underlying dynamics
        dt = TtM[t] - TtM[t-1]
        S_t = S_prev * exp((Drift[t] - 0.5 * Vol[t] ** 2) * dt + Vol[t] * RND[t] * sqrt(dt))
        # update previous value
        S_prev = S_t

        # kick out barrier touched at t
        if S_t >= S_k:
            return (1 + TtM[t] * I) * exp(- Disc[t] * TtM[t])

    # kick out barrier never touched before the maturity
    if S_t > S_p:
        return (exp(- Disc[t] * TtM[t]))
    else:
        return (S_t / S_0 * exp(- Disc[t] * TtM[t]))

# modified function for distribuited monte carlo method
def _payoff(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I):

    # INPUT:
    # t_steps : time steps
    # TtM     : time to maturity
    # Drift   : drift list by time
    # Vol     : volatility list by time
    # Disc    : discount rate
    # S_0     : underlying initial value
    # S_k     : kickout barrier
    # S_p     : protection barrier
    # N       : nominal value
    # I       : yearly interest over the nominal

    # OUTPUT:
    # out     : autocallable structure simulated discounted payoff

    from numpy import exp, sqrt, random

    # vars
    S_prev = S_0
    TtM_prev = 0

    # simu
    for t in range(len(t_steps)):

        # underlying dynamics
        S_t = S_prev * exp((Drift[t] - 0.5 * Vol[t] ** 2) * (TtM[t] - TtM_prev) + Vol[t] * random.randn() * sqrt(TtM[t] - TtM_prev))

        # update previous values
        TtM_prev = TtM[t]
        S_prev = S_t

        # kick out barrier touched at t
        if S_t >= S_k:
            return (1 + TtM[t] * I) * exp(- Disc[t] * TtM[t])

    # kick out barrier never touched before the maturity
    if S_t > S_p:
        return (exp(- Disc[t] * TtM[t]))
    else:
        return (S_t / S_0 * exp(- Disc[t] * TtM[t]))


# PRICING TOOLS :
# * classic monte carlo method
# * parallel monte carlo method
# * distribuited monte carlo method with pyspark

def monteCarloPrice(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, n_simu, RND):

    # INPUT:
    # t_steps : time steps
    # TtM     : time to maturity
    # Drift   : drift list by time
    # Vol     : volatility list by time
    # Disc    : discount rate
    # S_0     : underlying initial value
    # S_k     : kickout barrier
    # S_p     : protection barrier
    # N       : nominal value
    # I       : yearly interest over the nominal
    # n_simu  : number of simulations
    # RND     : random terms

    # OUTPUT:
    # out     : autocallable structure price

    import numpy as np 
    import pandas as pd 
    import time

    if RND == None:
        # generate pseudo-random sequence
        RND = pd.DataFrame(np.random.randn(int(n_simu), len(t_steps)), columns=t_steps)

    # starting_t = time.time()

    payoffs = [0] * int(n_simu) # initializes list
    for i in range(int(n_simu)):
        payoffs[i] = payoff(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, RND.iloc[i])
    
    # elapsed_t = time.time() - starting_t 
    # print('\nMonte Carlo simulation completed in', elapsed_t, 's')

    return sum(payoffs) / n_simu


# using Parallel from joblib package
def parallelMonteCarloPrice(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, n_simu, RND):

    # INPUT:
    # t_steps : time steps
    # TtM     : time to maturity
    # Drift   : drift list by time
    # Vol     : volatility list by time
    # Disc    : discount rate
    # S_0     : underlying initial value
    # S_k     : kickout barrier
    # S_p     : protection barrier
    # N       : nominal value
    # I       : yearly interest over the nominal
    # n_simu  : number of simulations
    # RND     : random terms

    # OUTPUT:
    # out     : autocallable structure price

    from joblib import Parallel, delayed
    import numpy as np 
    import pandas as pd 
    import time

    if RND == None:
        # generate pseudo-random sequence
        RND = pd.DataFrame(np.random.randn(int(n_simu), len(t_steps)), columns=t_steps)

    # starting_t = time.time()
    
    payoffs = [0] * int(n_simu) # initializes list

    # 'n_jobs=-1' uses all the PCU cores
    payoffs = Parallel(n_jobs=-1)(delayed(payoff)(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, RND.iloc[i]) for i in range(int(n_simu)))

    # elapsed_t = time.time() - starting_t
    # print('\nMonte Carlo simulation completed in', elapsed_t, 's')

    return sum(payoffs) / n_simu


# using PySpark
def distribuitedMonteCarloPrice(inputParameter, flagParameter, t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I, sc, n_simu):

    # INPUT:
    # t_steps : time steps
    # TtM     : time to maturity
    # Drift   : drift list by time
    # Vol     : volatility list by time
    # Disc    : discount rate
    # S_0     : underlying initial value
    # S_k     : kickout barrier
    # S_p     : protection barrier
    # N       : nominal value
    # I       : yearly interest over the nominal
    # sc      : spark context
    # n_simu  : number of simulations

    # OUTPUT:
    # out     : autocallable structure price

    import numpy as np 
    import pandas as pd 
    import time
    import pyspark
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row

    # convert pandas series to numpy array
    TtM = TtM.to_numpy()
    Drift = Drift.to_numpy()
    Vol = Vol.to_numpy()
    Disc = Disc.to_numpy()
    # initialize accumulator
    outputSum = sc.accumulator(0)
    
    if flagParameter == []:
        def sparkCustomizedPayoff(_input):
            outputSum.add(_payoff(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, I))
    elif flagParameter == 'V':
        def sparkCustomizedPayoff(_input):
            _volatility = Vol * _input
            outputSum.add(_payoff(t_steps, TtM, Drift, _volatility, Disc, S_0, S_k, S_p, N, I))
    elif flagParameter == 'D':
        def sparkCustomizedPayoff(_input):
            _spot = S_0 * _input
            outputSum.add(_payoff(t_steps, TtM, Drift, Vol, Disc, _spot, S_k, S_p, N, I))
    elif flagParameter == 'R':
        def sparkCustomizedPayoff(_input):
            _interest = I * _input
            outputSum.add(_payoff(t_steps, TtM, Drift, Vol, Disc, S_0, S_k, S_p, N, _interest))
    else:
        return 0


    # generate python collection of initial variables
    inputData = [inputParameter] * n_simu

    # create RDD
    simuRDD = sc.parallelize(inputData)
    # print('RDD size :', simuRDD.count()) # debug

    # distribuited computation
    simuRDD.foreach(sparkCustomizedPayoff)
    return outputSum.value / n_simu


def startDistribuitedEnvironment():
    
    # import and initialize spark
    import findspark
    findspark.init()
    import pyspark
    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setAppName('mySparkApp')
    sc = SparkContext(conf=conf)
    # spark_context.setLogLevel('WARN')
    
    print('Initialized PySpark, Conf and Context.\n') # debug
    return sc