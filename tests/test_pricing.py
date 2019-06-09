import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil.esg import ESG
from esglil.ir_models.hw1f_annual_exact import (ShortRate, CashAccount,
                                                BondPrice)
from esglil.ir_models.hw1f_annual_exact import (get_g_function, get_h_function,
                                                calibrate_b_function, calc_r_zero)
from esglil.pricing_models.black_scholes import BlackScholeEuropeanPrice
from esglil.pricing_models.hw1f import HW1fSwaptionPrice
from esglil.equity_models import GeometricBrownianMotion

import numpy as np

class hw1f_test_swaption_price(unittest.TestCase):
    def setUp(self):
        #basic swaption information
        maturity = 10
        tenor = 5
        #esg basic elements
        bond_prices = {t: 1.01**(-t) for t in range(1,41)}
        delta_t = 1
        Z = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        alpha = 0.001
        sigma = 0.01
        b_vec = calibrate_b_function(bond_prices, alpha, sigma)
        b_fc = lambda t: b_vec[int(t)]
        g = get_g_function(b_s=b_fc, alpha=alpha)
        h = get_h_function(b=b_fc, alpha=alpha)
        r = ShortRate(g=g, sigma_hw=sigma,
                              alpha_hw=alpha, r_zero=0.02, Z=Z)
        #bonds
        bonds_dict1 = {}
        bonds_dict2 = {}
        for t in range(maturity, maturity+tenor+2):
            P = BondPrice(alpha=alpha, r=r, sigma_hw=sigma,
                                 h=h, T=t)
            bonds_dict1['P{}'.format(t)] = P
            bonds_dict2[t] = P
        #swaption
        strike = 0.01
        volatility = 0.1
        alpha = 0.01
        swaption = HW1fSwaptionPrice(maturity, tenor, strike, volatility, alpha,
                                     bonds_dict=bonds_dict2, swpt_type='payer')

        #put together esg
        self.esg = ESG(dt_sim=delta_t, Z=Z, r=r, **bonds_dict1, swaption=swaption)

    def test_shape(self):
        results = self.esg.run_multistep_to_dict(dt_out=1, max_t=8)
        self.assertEqual(type(results), dict,
                         'incorrect type')
        self.assertEqual(len(results), 8)
        self.assertTrue('swaption' in results[1])



class BlackScholesTestOptionPrice(unittest.TestCase):
    def setUp(self):
        #basic swaption information
        maturity = 10
        #esg basic elements
        bond_prices = {t: 1.01**(-t) for t in range(1,41)}
        delta_t = 1
        rho = 0.2
        corr = [[1, rho], [rho, 1]]
        C = np.diag([delta_t, delta_t])
        cov = C*corr*C.T
        dW = rng.NormalRng(dims=2, sims=1000, mean=[0, 0], cov=cov)
        hw_alpha = 0.001
        hw_sigma = 0.01
        b_vec = calibrate_b_function(bond_prices, hw_alpha, hw_sigma)
        b_fc = lambda t: b_vec[int(t)]
        g = get_g_function(b_s=b_fc, alpha=hw_alpha)
        h = get_h_function(b=b_fc, alpha=hw_alpha)
        r = ShortRate(g=g, sigma_hw=hw_sigma,
                              alpha_hw=hw_alpha, r_zero=0.02, Z=dW[0])
        volatility = 0.1
        S = GeometricBrownianMotion(mu=r, sigma=volatility, dW=dW[1])
        #bonds
        P = BondPrice(alpha=hw_alpha, r=r, sigma_hw=hw_sigma,
                             h=h, T=maturity)

        #call
        strike = 200
        volatility = 0.1
        option = BlackScholeEuropeanPrice(maturity, strike, volatility,
                                     opt_type='call', underlying=S,
                                          maturity_bond=P)

        #put together esg
        self.esg = ESG(dt_sim=delta_t, dW=dW, r=r, bond=P, S=S, option=option)

    def test_shape(self):
        results = self.esg.run_multistep_to_dict(dt_out=1, max_t=10)
        # print([results[time]['option'].mean() for time in results])
        self.assertEqual(type(results), dict,
                         'incorrect type')
        self.assertEqual(len(results), 10)
        self.assertTrue('option' in results[1])


if __name__ == '__main__':
    unittest.main()