def interestRate(Call, Put, Spot, Strk, TtM):

    # INPUT
    # Call : call price
    # Put  : put price
    # Spot : spot price
    # Strk : strike price
    # TtM  : time to maturity

    # OUTPUT
    # out  : interest rate

    import numpy as np 

    return np.log(Strk / (Spot + Put - Call)) / TtM


def impliedVolatility(target, flag, spot, strk, ttM, inR):

    # INPUT: 
    # target : target price (Call or Put)
    # flag   : 'C' if call option 'P' if put option
    # spot   : spot price
    # strk   : strike price
    # ttM    : time to maturity
    # inR    : interest rate

    # OUTPUT:
    # sigma  : implied volatility

    import myblackscholes as bs
    import numpy as np
    import pandas as pd 

    MAX_ITER = 100
    MAX_ERROR = 1.0e-5

    sigma = 0.5
    i = 0
    while (i < MAX_ITER):
        if flag == 'C': # call
            price = bs.callPrice(spot, strk, ttM, inR, sigma)
        elif flag == 'P': # put
            price = bs.putPrice(spot, strk, ttM, inR, sigma)
        else: # error
            price = target * 0 

        vega = bs.vega(spot, strk, ttM, inR, sigma)
        diff = target - price

        if (abs(diff) < MAX_ERROR):
            return sigma # vol

        sigma = sigma + diff / vega
        i += 1

    # return best value so far
    return sigma


def impliedVolatilitySurface(Target, flag, Spot, Strk, TtM, InR):

    # INPUT: 
    # target : target price (Call or Put)
    # opt    : 'C' if call option 'P' if put option
    # Spot   : spot price
    # Strk   : strike price
    # TtM    : time to maturity
    # InR    : interest rate

    # OUTPUT:
    # Sigma  : implied volatility surface
    
    import myblackscholes as bs
    import numpy as np
    import pandas as pd 

    # initializes dataframe at 0.5
    Sigma_col = Target.columns
    Sigma = pd.DataFrame(0.5 * np.ones(Target.shape), columns=Sigma_col)
    for i in range(0, Sigma.shape[0]):
        for t in Sigma_col:
            Sigma[t][i] = impliedVolatility(Target[t][i], flag, Spot[t][i], Strk[t][i], TtM[t][i], InR[t][i])

    return Sigma
    

'''
def parallelImpliedVolatilitySurface(Target, opt_flag, Spot, Strk, TtM, InR):

    # INPUT: 
    # target : target price (Call or Put)
    # opt_flag : 'C' if call option 'P' if put option
    # Spot   : spot price
    # Strk   : strike price
    # TtM    : time to maturity
    # InR    : interest rate

    # OUTPUT:
    # Sigma  : implied volatility

    import myblackscholes as bs
    import numpy as np
    import pandas as pd 

    MAX_ITER = 100
    MAX_ERROR = 1.0e-5

    Sigma = pd.DataFrame(0.5 * np.ones(Target.shape), columns=Target.columns)
    FLAG = (Sigma == 0.5)
    print(FLAG)
    Price = Target * 0
    Vega = Target * 0

    i = 0
    while (i < MAX_ITER):
        if opt_flag == 'C': # call
            Price[FLAG] = bs.callPrice(Spot[FLAG], Strk[FLAG], TtM[FLAG], InR[FLAG], Sigma[FLAG])
        elif opt_flag == 'P': # put
            Price = bs.putPrice(Spot, Strk, TtM, InR, Sigma)
        else: # error
            return Target * 0

        Vega[FLAG] = bs.vega(Spot[FLAG], Strk[FLAG], TtM[FLAG], InR[FLAG], Sigma[FLAG])
        Diff = Target - Price
        FLAG = Diff.abs() > MAX_ERROR

        # debug
        print('i =', i)
        print(Diff)

        if (FLAG.any(axis=None) == False):
            return Sigma # vol

        Sigma[FLAG] = Sigma[FLAG] + Diff[FLAG] / Vega[FLAG]
        i += 1

    # return best value so far
    return Sigma

'''