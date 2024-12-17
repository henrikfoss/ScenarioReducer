from typing import Any
import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW
from ScenarioReducer import Fast_forward
from ScenarioReducer.fast_forward import compute_distance_matrix, redistribute_probs

'''
@author: pdb567
@date:11/05/2022

Run from cmd with print option: pytest -s test_fast_forward.py
'''

class NormalScenarioGenerator:
    def __init__(self, loc=0, scale=1, seed=47):
        self.loc = loc
        self.scale = scale
        self.seed = seed

    def make_scenarios(self, n_scenarios=500, seed=None):
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed)
        y = rng.normal(self.loc, self.scale, (1, n_scenarios))
        return y
    
    @property
    def rv(self):
        return norm(self.loc, self.scale)
    
    def __getattr__(self, __name: str) -> Any:
        # Forward other attributes to rv
        return self.rv.__getattribute__(__name)


@pytest.fixture(scope="module")
def scenario_generator():
    gen = NormalScenarioGenerator()
    return gen


def test_redistribute_probs():
    # redistribute_probs(indxR, probs_initial, dist_mtrx, J_set)
    N = 10
    y = np.arange(N, dtype=float).reshape((1, -1))
    indxR = np.array([2, 6])
    probs_initial = np.ones(N) / N
    dist_mtrx = compute_distance_matrix(y, np.inf)
    J_set = np.setdiff1d(np.arange(N), indxR)
    print(J_set)
    probs_reduced = redistribute_probs(indxR, probs_initial, dist_mtrx, J_set)
    np.testing.assert_equal(probs_reduced, np.array([5, 5])/N)


def test_fast_forward_stats(scenario_generator):
    """
    Verify that reduced scenario set has similar statistical properties to
    the original scenario set.
    """
    n_scenarios_original = 1000
    scenario_reduction_factor = 5

    y = scenario_generator.make_scenarios(n_scenarios_original)
    FFreducer = Fast_forward(y, np.ones(y.shape[1])/y.shape[1])
    y_reduced, y_reduced_probs = FFreducer.reduce(np.inf, n_scenarios_original // scenario_reduction_factor)
    
    d_original = DescrStatsW(data=y.T)
    d_reduced = DescrStatsW(data=y_reduced.T, weights=y_reduced_probs)

    print_comparative_stats(d_original, d_reduced)

    # Since we are dealing with random samples, tolerances cannot be too tight.
    # Using larger scenario sets will be more accurate, but then the test takes a long time.
    np.testing.assert_allclose(d_original.mean, d_reduced.mean, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(d_original.std, d_reduced.std, rtol=1e-2, atol=1e-4)
    q = [0.05, 0.5, 0.95]
    np.testing.assert_allclose(d_original.quantile(q), d_reduced.quantile(q), rtol=1e-2, atol=1e-3)

def test_fixed_scenarios():
    outcomes_original = np.array([[215., 216., 217., 218., 219., 220., 221., 222., 223., 224., 225.,
        226., 227., 228., 229., 230., 231., 232., 233., 234., 235., 236.,
        237., 238., 239., 240., 241., 242., 243., 244., 245., 246., 247.,
        248., 249., 250., 251., 252., 253., 254., 255., 256., 257., 258.,
        259., 260., 261., 262., 263., 264., 265., 266., 267., 268., 269.,
        270., 271., 272., 273., 274., 275., 276.]])

    probabilities_original = np.array([0.00419851, 0.00476228, 0.00537685, 0.00604288, 0.0067604,
           0.00752875, 0.00834648, 0.00921134, 0.01012024, 0.01106919,
           0.0120533, 0.01306683, 0.01410319, 0.01515497, 0.01621409,
           0.01727179, 0.01831885, 0.01934564, 0.02034229, 0.02129888,
           0.02220556, 0.02305274, 0.02383126, 0.02453255, 0.02514882,
           0.02567315, 0.02609966, 0.02642362, 0.02664151, 0.02675111,
           0.02675152, 0.02664319, 0.02642786, 0.02610857, 0.02568955,
           0.02517615, 0.02457471, 0.02389245, 0.02313731, 0.02231783,
           0.02144295, 0.02052189, 0.01956397, 0.01857848, 0.01757451,
           0.01656085, 0.01554586, 0.01453739, 0.01354264, 0.01256818,
           0.01161981, 0.01070263, 0.00982091, 0.00897821, 0.00817731,
           0.00742026, 0.00670846, 0.00604264, 0.00542297, 0.00484908,
           0.00432015, 0.00383498])

    FFreducer = Fast_forward(initialSet=outcomes_original, initProbs=probabilities_original)

    test_cases_fixed_idx = [
        [1,2,3],
        [5],
        [4,10],
        [0, len(outcomes_original)-1],
        [0,1,2,3,4,5,6,7,8,9,10]
    ]

    for fixed_idx in test_cases_fixed_idx:
        outcomes_reduced, probabilities_reduced = FFreducer.reduce(
            np.inf, 10, fixed_idx=fixed_idx
        )
        assert np.isclose(probabilities_reduced.sum(), 1.0)
        for idx in fixed_idx:
            assert outcomes_original[idx] in outcomes_reduced


def print_comparative_stats(d1, d2):
    compare_list = {
        'd1': d1,
        'd2': d2,
    }

    stats = [
        ('mean',),
        ('std',),
        ('quantile', 0.05),
        ('quantile', 0.5),
        ('quantile', 0.95),
    ]

    for s in stats:
        if len(s) == 1:
            s_txt = s[0]
            for descr, d in compare_list.items():
                print(f'{descr:10} {s_txt} = {d.__getattribute__(s[0]).item():.5f}')
        else:
            s_txt = f'{s[0]}({s[1]})'
            for descr, d in compare_list.items():
                print(f'{descr:10} {s_txt} = {d.__getattribute__(s[0])(s[1], return_pandas=False).item():.5f}')
