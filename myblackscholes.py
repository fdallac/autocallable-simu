def callPrice(Spot, Strk, TtM, InR, Sigma):
    
    # INPUT
    # Spot  : spot price
    # Strk  : strike price
    # TtM   : time to maturity
    # InRn  : interest rate
    # Sigma : volatility

    # OUTPUT
    # out   : call price

    from numpy import log, exp, sqrt
    from scipy.stats import norm
    
    d1 = (log(Spot / Strk) + (InR + 0.5 * Sigma ** 2) * TtM) / (Sigma * sqrt(TtM))
    d2 = (log(Spot / Strk) + (InR - 0.5 * Sigma ** 2) * TtM) / (Sigma * sqrt(TtM))
    
    return (Spot * norm.cdf(d1, 0.0, 1.0) - Strk * exp(-InR * TtM) * norm.cdf(d2, 0.0, 1.0))


def putPrice(Spot, Strk, TtM, InR, Sigma):
     
    # INPUT
    # Spot  : spot price
    # Strk  : strike price
    # TtM   : time to maturity
    # InRn  : interest rate
    # Sigma : volatility

    # OUTPUT
    # out   : call price
   
    from numpy import log, exp, sqrt
    from scipy.stats import norm
    
    d1 = (log(Spot / Strk) + (InR + 0.5 * Sigma ** 2) * TtM) / (Sigma * sqrt(TtM))
    d2 = (log(Spot / Strk) + (InR - 0.5 * Sigma ** 2) * TtM) / (Sigma * sqrt(TtM))

    return (Strk * exp(-InR * TtM) * norm.cdf(-d2, 0.0, 1.0) - Spot * norm.cdf(-d1, 0.0, 1.0))


def vega(Spot, Strk, TtM, InR, Sigma):
    
    # INPUT
    # Spot  : spot price
    # Strk  : strike price
    # TtM   : time to maturity
    # InRn  : interest rate
    # Sigma : volatility

    # OUTPUT
    # out   : call price

    from scipy.stats import norm
    from numpy import log, sqrt

    d1 = (log(Spot / Strk) + (InR + 0.5 * Sigma ** 2) * TtM) / (Sigma * sqrt(TtM))

    return Spot * sqrt(TtM) * norm.cdf(d1)