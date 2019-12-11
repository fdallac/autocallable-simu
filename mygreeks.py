def computePricesForGreek(greekFlag, t_steps, TtM, Drift, Vol, DsR, S_0, S_k, S_p, N, I, r_param, n_simu, sc):
    # INPUT:
    # greekFlag : 'V' = Vega
    #             'D' = Delta
    #             'R' = Rho
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
    # r_param : range of the parameters
    # n_simu  : number of simulation by monte carlo
    # sc      : spark context

    # OUTPUT:
    # prices  : list of prices
    # denoms  : parameter over wich the price is computed

    import myautocallable as acl

    prices = []
    denoms = []

    counter = 0
    for param in r_param:
        print(int(counter / len(r_param) * 100), '% of simulation')
        prices.append(acl.distribuitedMonteCarloPrice(param, greekFlag, t_steps, TtM, Drift, Vol, DsR, S_0, S_k, S_p, N, I, sc, n_simu))
        if greekFlag == 'V':
            denoms.append(Vol.mean() * param)
        elif greekFlag == 'D':
            denoms.append(S_0 * param)
        elif greekFlag == 'R':
            denoms.append(Drift.mean() * param)
        else:
            print('ERROR')
            return 0, 0
        counter += 1

    print(int(counter / len(r_param) * 100), '% of simulation')
    return prices, denoms


def derivateGreek(num, denom):
    
    # INPUT:
    # num : numerator of the differential operator
    # denom : denominator of the differential operator

    # OUTPUT
    # greek : values of the derivatives over denom_prime
    # denom_gr : will be len(denom) - 1

    greek = []
    denom_gr = []
    for i in range(len(num)-1):
        greek.append((num[i+1] - num[i]) / (denom[i+1] - denom[i]))
        denom_gr.append((denom[i+1] + denom[i]) / 2)

    return greek, denom_gr


def plot(greek, denom_gr, primitive, denom_pr, label_gr, label_pr, label_denom):

    import matplotlib.pyplot as plt 

    # plot vega
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax1.set_xlabel(label_denom)
    ax1.set_ylabel(label_pr, color='tab:blue')
    ax1.plot(denom_pr, primitive, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_ylabel(label_gr, color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(denom_gr, greek, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig, ax1, ax2