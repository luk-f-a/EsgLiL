import numpy as np
from esglil.common import StochasticVariable

try:
    import QuantLib as ql
except ModuleNotFoundError:
    pass
except:
    raise

class BlackScholeEuropeanPrice(StochasticVariable):
    """class for pricing European Swaptions (cash settled) using a
    1F HW model with Jamshidian decomposition

     Parameters
    ----------

    maturity: integer
        maturity (number of years between beginning of simulation and payment)

    tenor: integer
        tenor or underlying swap in years

    bonds_dict:  dictionary or mapping object
        dictionary of BondPrice objects. These objects need to be registered
        with the Model, so that their price is updated.
        The dictionary must have entries for every integer between
        maturity and tenor + 1, both endpoints included.
    """

    __slots__ = ['mat_bond', 'maturity', 'rate', 'ql_option', 'ql_date',
                 'calendar', 'underlying', 'underlyingQuote' ]

    def __init__(self, maturity, strike, flat_vol, opt_type,
                 underlying, maturity_bond):
        assert opt_type in ('call', 'put')
        self.maturity = maturity
        self.underlying = underlying
        ql_opt_type = ql.Option.Call if opt_type == 'call' else ql.Option.Put
        calendar = ql.TARGET()
        self.calendar = calendar

        #the today date is irrelevant but QuantLib needs to build a calendar
        today = ql.Date(30, ql.December, 2018)
        self.ql_date = today
        maturity_date = calendar.advance(today, maturity, ql.Years)
        ql.Settings.instance().evaluationDate = today
        # as long as always working with annual rates, the day count should be irrelevant
        day_count = ql.Actual365Fixed()
        self.mat_bond = maturity_bond
        ir_rate = ql.SimpleQuote(maturity_bond ** (-1 / maturity) - 1)
        self.rate = ir_rate
        rateHandle = ql.QuoteHandle(ir_rate)
        riskFreeRate = ql.FlatForward(today, rateHandle, day_count)
        discount_curve = ql.YieldTermStructureHandle(riskFreeRate)

        # surface
        volatility = ql.BlackConstantVol(0, calendar, flat_vol, day_count)

        # option parameters
        exercise = ql.EuropeanExercise(maturity_date)
        payoff = ql.PlainVanillaPayoff(ql_opt_type, strike)

        # market data
        underlyingQuote = ql.SimpleQuote(underlying.value_t)
        self.underlyingQuote = underlyingQuote
        divRate = 0
        dividendYield = ql.FlatForward(today, divRate, day_count)
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlyingQuote),
                                               ql.YieldTermStructureHandle(dividendYield),
                                               discount_curve,
                                               ql.BlackVolTermStructureHandle(volatility))

        option = ql.EuropeanOption(payoff, exercise)
        engine = ql.AnalyticEuropeanEngine(process)

        # method: analytic
        option.setPricingEngine(engine)
        self.ql_option = option

        self.value_t = None
        self.t_1 = 0

    def run_step(self, t):
        self.ql_date = self.calendar.advance(self.ql_date, int(t - self.t_1),
                                             ql.Years)
        if t == 1:
            #to avoid options having 0 value at maturity due to the
            #esglil maturity falling 1 day after the QL maturity
            self.ql_date = self.ql_date - ql.Period(2, ql.Days)
        ql.Settings.instance().evaluationDate = self.ql_date

        self.value_t = np.zeros(self.mat_bond.value_t.shape[-1])
        new_rate_val = self.mat_bond ** (-1 / self.maturity) - 1
        new_underlying_val = self.underlying.value_t
        for i in range(len(self.value_t)):
            self.underlyingQuote.setValue(new_underlying_val[i])
            self.rate.setValue(new_rate_val[i])
            self.value_t[i] = self.ql_option.NPV()
        self.t_1 = t