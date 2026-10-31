import copy
import math
import unittest
from benchmark_sql import compare, quality, FROZEN


def sample(variant):
    return dict(variant=variant, commit=FROZEN if variant=='baseline' else 'a'*40,
                dataset='SIFT250K', rows=250000, dim=128, nq=4096,
                inputs=dict(base='basehash',queries='queryhash',truth='truthhash'),
                build_seconds=20., queries={str(k):dict(qps=100.,recall=.99,integrity=True) for k in (10,100)})


class BenchmarkTests(unittest.TestCase):
    def test_integrity_checks_count_duplicates_and_id_domain(self):
        for found in ([1,1], [1], [1,9]):
            self.assertFalse(quality([found], [[1,2]], {1,2,3}, 2)['integrity'])
        self.assertEqual(quality([[2,1]], [[1,2]], {1,2,3}, 2), dict(integrity=True,recall=1.))
        self.assertEqual(quality([[1,3]], [[1,2]], {1,2,3}, 2)['recall'], .5)

    def test_reject_invalid_truth(self):
        for actual, truth in (([[1]], []), ([[1]], [[1,1]]), ([], [])):
            with self.assertRaises(ValueError):
                quality(actual, truth, {1,2}, 2)

    def test_compare_pairs_and_geometric_mean(self):
        a,b=sample('baseline'),sample('candidate')
        b['build_seconds']=10.
        b['queries']['10']['qps']=200.
        b['queries']['100']['qps']=50.
        result=compare(a,b)
        self.assertEqual(result['build_speedup'],2.)
        self.assertEqual(result['query_speedup'],1.)
        self.assertEqual(result['measured_gate_failures'],[])
        self.assertIsNone(result['official_score'])

    def test_gate_boundaries_and_failure(self):
        a,b=sample('baseline'),sample('candidate')
        for k in ('10','100'):
            b['queries'][k].update(qps=90.,recall=.985)
        self.assertEqual(compare(a,b)['measured_gate_failures'],[])
        b['queries']['10'].update(qps=89.,recall=.94,integrity=False)
        gates=compare(a,b)['measured_gate_failures']
        for gate in ('result_integrity@10','recall@10_minimum','recall@10_baseline_delta','query_regression'):
            self.assertIn(gate,gates)

    def test_reject_unpaired_or_nonfinite_measurements(self):
        a,b=sample('baseline'),sample('candidate')
        b['inputs']['base']='different'
        with self.assertRaises(ValueError): compare(a,b)
        for value in (0., -1., math.inf, math.nan):
            b=sample('candidate');b['queries']['10']['qps']=value
            with self.assertRaises(ValueError): compare(a,b)
        a['commit']='0'*40
        with self.assertRaises(ValueError): compare(a,sample('candidate'))

    def test_hidden_thresholds_and_top100_delta(self):
        a,b=sample('baseline'),sample('candidate')
        a['dataset']=b['dataset']='Hidden250K'
        a['queries']['10']['recall']=.90
        a['queries']['100']['recall']=.85
        b=copy.deepcopy(a);b['variant']='candidate'
        self.assertEqual(compare(a,b)['measured_gate_failures'],[])
        b['queries']['100']['recall']=.849
        self.assertIn('recall@100_minimum',compare(a,b)['measured_gate_failures'])


if __name__=='__main__': unittest.main()
