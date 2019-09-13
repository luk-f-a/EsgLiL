import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from esglil import model_zoo


class test_hw1f_annual(unittest.TestCase):
    def test_shape(self):
        n = 2
        bond_prices = {t:1.01**(-t) for t in range(0,100)}
        esg = model_zoo.get_hw_gbm_annual(sims=n, rho=0.5, bond_prices=bond_prices,
                                          hw_alpha=0.5,
                                          hw_sigma=0.01, gbm_sigma=0.2,
                                          const_tau=None,
                                          out_bonds=None, custom_ind_z=None,
                                          seed=None)
        d_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=10,
                                               out_vars=['cash', 'r', 'S'])
        self.assertTrue('cash' in d_full_run[1])
        self.assertTrue('r' in d_full_run[1])
        self.assertTrue('S' in d_full_run[1])
        self.assertTrue(10 in d_full_run)
        self.assertTrue(d_full_run[1]['cash'].shape == (n,))


if __name__ == '__main__':
    unittest.main()