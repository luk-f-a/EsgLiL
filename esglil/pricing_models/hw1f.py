import numpy as np
from esglil.common import FunctionOfVariable

try:
    import QuantLib as ql
except ModuleNotFoundError:
    pass
except:
    raise


class HW1fSwaptionPrice(FunctionOfVariable):
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

    __slots__ = ['bonds_dict', 'rates', 'ql_swaption', 'ql_date', 'calendar',
                 'notional']

    def __init__(self, maturity, tenor, fixed_rate, volatility, alpha, swpt_type,
                 bonds_dict):
        assert swpt_type in ('receiver', 'payer')
        calendar = ql.TARGET()
        self.calendar = calendar

        #the today date is irrelevant but QuantLib needs to build a calendar
        today = ql.Date(30, ql.December, 2018)
        self.ql_date = today
        effective = calendar.advance(today, maturity, ql.Years)
        maturity_date = calendar.advance(effective, tenor, ql.Years)
        ql.Settings.instance().evaluationDate = today
        # as long as always working with annual rates, the day count should be irrelevant
        day_count = ql.Actual365Fixed()
        rates = {time: ql.SimpleQuote(bonds_dict[time]**(-1/time)-1)
                 for time in range(maturity, maturity + tenor + 2)}
        self.rates = rates
        self.bonds_dict = bonds_dict
        settlementDays = 0
        settlementDate = calendar.advance(today, settlementDays, ql.Days)

        rateHelpers = [ql.DepositRateHelper(ql.QuoteHandle(rates[time]),
                                            ql.Period(time, ql.Years),
                                            settlementDays,
                                            calendar, ql.ModifiedFollowing,
                                            False, day_count)
                       for time in range(maturity, maturity + tenor + 2)]
        depoCurve = ql.PiecewiseFlatForward(settlementDate, rateHelpers, day_count)
        discount_curve = ql.YieldTermStructureHandle(depoCurve)
        libor_1y = ql.USDLibor(ql.Period('1Y'), discount_curve)


        fixed_schedule = ql.Schedule(effective, maturity_date, ql.Period('1Y'),
                                     calendar, ql.ModifiedFollowing,
                                     ql.ModifiedFollowing,
                                     ql.DateGeneration.Forward, False)
        float_schedule = ql.Schedule(effective, maturity_date, ql.Period('1Y'),
                                     calendar, ql.ModifiedFollowing,
                                     ql.ModifiedFollowing,
                                     ql.DateGeneration.Forward, False)
        self.notional = 1
        fixed_rate = fixed_rate
        swap_type = {'payer': ql.VanillaSwap.Payer,
                     'receiver': ql.VanillaSwap.Receiver}[swpt_type]
        swap = ql.VanillaSwap(swap_type, self.notional, fixed_schedule,
                              fixed_rate, day_count, float_schedule, libor_1y,
                              0., day_count)
        # Swaption definition
        self.ql_swaption = ql.Swaption(swap, ql.EuropeanExercise(effective),
                                       ql.Settlement.Cash,
                                       ql.Settlement.CollateralizedCashPrice)
        model = ql.HullWhite(discount_curve, sigma=volatility, a=alpha)
        engine = ql.JamshidianSwaptionEngine(model)
        self.ql_swaption.setPricingEngine(engine)

        self.value_t = None
        self.t_1 = 0

    def run_step(self, t):
        self.ql_date = self.calendar.advance(self.ql_date, int(t - self.t_1), ql.Years)
        if t == 1:
            #to avoid swaptions having 0 value at maturity due to the
            #esglil maturity falling 1 day after the QL maturity
            self.ql_date = self.ql_date - ql.Period(2, ql.Days)
        ql.Settings.instance().evaluationDate = self.ql_date
        first_t = list(self.rates.keys())[0]
        self.value_t = np.zeros(self.bonds_dict[first_t].value_t.shape[-1])
        new_rates_val = {time: self.bonds_dict[time] ** (-1 / time) - 1
                         for time in self.rates}

        for i in range(len(self.value_t)):
            for time in self.rates:
                new_rate_val = new_rates_val[time].squeeze()[i]
                self.rates[time].setValue(new_rate_val)
            self.value_t[i] = self.ql_swaption.NPV()
        self.t_1 = t
