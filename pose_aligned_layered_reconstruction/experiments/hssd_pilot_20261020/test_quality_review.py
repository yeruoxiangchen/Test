"""A sampling review must never cherry-pick seeds or waive coordinate failures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location('hssd_quality_review_test', HERE / 'quality_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def valid_record():
    return {'hard_checks': {key: True for key in review.HARD_CHECKS},
            'samples': [{'seed': seed, 'count': 100000,
                         'surface': {'chamfer_l1': .01, 'fscore_0p02': .90}}
                        for seed in [42, 43, 44]]}


class QualityReviewTests(unittest.TestCase):
    def test_all_seeds_must_pass_not_only_mean_or_best(self):
        record = valid_record()
        self.assertTrue(review.record_passes(record))
        record['samples'][0]['surface']['fscore_0p02'] = .8495549485879796
        self.assertFalse(review.record_passes(record))

    def test_each_seed_must_also_pass_chamfer(self):
        record = valid_record()
        record['samples'][2]['surface']['chamfer_l1'] = .015001
        self.assertFalse(review.record_passes(record))

    def test_missing_or_duplicate_seed_is_rejected(self):
        for seeds in [[42, 43], [42, 42, 44], [43, 42, 44]]:
            record = valid_record()
            record['samples'] = [dict(copy.deepcopy(record['samples'][0]), seed=seed) for seed in seeds]
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, 'fixed seeds'):
                review.record_passes(record)

    def test_sample_density_cannot_change_per_object(self):
        record = valid_record()
        record['samples'][1]['count'] = 20000
        with self.assertRaisesRegex(ValueError, 'sample count'):
            review.record_passes(record)

    def test_surface_quality_cannot_override_nonstatistical_checks(self):
        for key in review.HARD_CHECKS:
            record = valid_record()
            record['hard_checks'][key] = False
            with self.subTest(check=key):
                self.assertFalse(review.record_passes(record))

    def test_empty_hard_checks_cannot_pass_vacuously(self):
        record = valid_record()
        record['hard_checks'] = {}
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            review.record_passes(record)

    def test_nonfinite_metrics_are_rejected(self):
        for value in [float('nan'), float('inf')]:
            record = valid_record()
            record['samples'][0]['surface']['chamfer_l1'] = value
            self.assertFalse(review.record_passes(record))

    def test_review_is_bound_to_artifacts_and_complete_original_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / 'geometry'
            artifact.write_text('unchanged')
            record = dict(valid_record(), uid='fixture', inputs=[review.binding(artifact)])
            report = {'settings': review.SETTINGS, 'objects': [record], 'passed': True,
                      **{key: review.binding(artifact) for key in
                         ['reviewer_source', 'metrics_source', 'plan', 'original_gate']}}
            report_file = root / 'review.json'
            report_file.write_text(json.dumps(report))
            with patch.object(review, 'REPORT', report_file), patch.object(
                    review, 'plan', return_value={'plan': {'objects': [{'uid': 'fixture'}]}}):
                self.assertTrue(review.validate_review()['passed'])
                artifact.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    review.validate_review()
                artifact.write_text('unchanged')
                report['objects'] = []
                report_file.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'omitted'):
                    review.validate_review()


if __name__ == '__main__':
    unittest.main()
