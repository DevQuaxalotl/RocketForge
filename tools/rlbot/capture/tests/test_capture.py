import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location('capture', Path(__file__).resolve().parents[1] / 'capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rf-capture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rows(self, indices=(0, 1)):
        records = [dict(kind='header', schema=capture.SCHEMA, indices=list(indices),
                        agent='synthetic', team=0, setup={}, checkpoint={'POLICY.lt': {'bytes': 0, 'crc32': 0}}),
                   dict(kind='initialize', team_info=None, field=None, match=None)]
        for callback in range(3):
            records.append(dict(kind='callback', callback=callback, rng='1', packet='0000', prediction=None,
                policy=[dict(index=i, action=0, observation=[i, 1.0], mask=[1, 0], probabilities=[1., 0.]) for i in indices],
                schedule=[dict(index=i, delay=3, policy_tick=callback) for i in indices],
                controls=[dict(index=i, values=[0.] * 8) for i in indices], inference_us=4.))
        records.append(dict(kind='footer', callbacks=3, lost=0, write_failed=False))
        return records

    def save(self, name, rows=None, partial=False):
        rows = self.rows() if rows is None else rows
        path = self.root / name
        with path.open('wb') as out:
            for seq, row in enumerate(rows):
                payload = json.dumps(dict(row, sequence=seq), separators=(',', ':'))
                out.write((json.dumps(dict(payload=payload, crc32=zlib.crc32(payload.encode()))) + '\n').encode())
            if partial:
                out.write(b'{"payload":')
        return path

    def test_complete_and_split_rows(self):
        ref = self.save('ref')
        a, b = self.save('a', self.rows((0,))), self.save('b', self.rows((1,)))
        report = capture.compare(ref, [b, a])
        self.assertEqual(report['status'], 'match')
        self.assertEqual(report['callbacks'], 3)
        self.assertEqual(report['candidate_inference']['calls'], 6)

    def test_reports_control_mask_and_observation_differences(self):
        rows = self.rows()
        rows[2]['policy'][0]['observation'][1] += .01
        rows[2]['policy'][0]['mask'] = [1, 1]
        rows[3]['controls'][1]['values'][0] = .5
        report = capture.compare(self.save('ref'), [self.save('changed', rows)])
        self.assertEqual(report['status'], 'different')
        self.assertEqual(set(report['differences']), {'observation', 'mask', 'controls'})

    def test_tolerances_do_not_relax_control_comparison(self):
        rows = self.rows()
        rows[2]['policy'][0]['observation'][1] += 1e-7
        ref, candidate = self.save('ref'), self.save('candidate', rows)
        self.assertEqual(capture.compare(ref, [candidate])['status'], 'match')
        rows[2]['controls'][0]['values'][0] = 1e-9
        self.assertEqual(capture.compare(ref, [self.save('control', rows)])['status'], 'different')

    def test_incomplete_prefix_is_never_a_complete_match(self):
        partial = self.save('partial', self.rows()[:-1], True)
        with self.assertRaisesRegex(capture.CaptureError, 'truncated'):
            capture.inspect(partial)
        self.assertFalse(capture.inspect(partial, True)['complete'])
        self.assertEqual(capture.compare(partial, [partial], True)['status'], 'prefix_match')
        self.assertEqual(capture.main(['inspect', str(partial), '--allow-incomplete']), 3)

    def test_corruption_is_not_salvaged(self):
        path = self.save('bad')
        data = path.read_bytes().replace(b'synthetic', b'changedxx')
        path.write_bytes(data)
        with self.assertRaisesRegex(capture.CaptureError, 'checksum'):
            capture.inspect(path, True)

    def test_sequence_gap_is_not_salvaged(self):
        path = self.save('gap')
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b''.join(lines[:2] + lines[3:]))
        with self.assertRaisesRegex(capture.CaptureError, 'sequence gap'):
            capture.inspect(path, True)

    def test_footer_loss_and_trailing_data(self):
        rows = self.rows()
        rows[-1]['lost'] = 1
        with self.assertRaisesRegex(capture.CaptureError, 'lost data'):
            capture.inspect(self.save('loss', rows), True)
        rows = self.rows() + [self.rows()[2]]
        with self.assertRaisesRegex(capture.CaptureError, 'after capture footer'):
            capture.inspect(self.save('trailing', rows))

    def test_missing_extra_and_different_inputs_fail(self):
        ref = self.save('ref')
        rows = self.rows()[:-2]
        with self.assertRaisesRegex(capture.CaptureError, 'missing callbacks'):
            capture.compare(ref, [self.save('short', rows)], True)
        rows = self.rows()
        rows[2]['packet'] = '1234'
        with self.assertRaisesRegex(capture.CaptureError, 'inputs differ'):
            capture.compare(ref, [self.save('input', rows)])
        with self.assertRaisesRegex(capture.CaptureError, 'exactly once'):
            capture.compare(ref, [ref, ref])

    def test_checkpoint_and_setup_must_match(self):
        ref = self.save('ref')
        for field in ('checkpoint', 'setup'):
            rows = self.rows()
            rows[0][field] = {'changed': True}
            with self.assertRaisesRegex(capture.CaptureError, field):
                capture.compare(ref, [self.save(field, rows)])

    def test_reject_invalid_diagnostics(self):
        for mutation in (lambda r: r['controls'].pop(),
                         lambda r: r['policy'][0].update(probabilities=[None, 0]),
                         lambda r: r['policy'][0].update(action=1),
                         lambda r: r['policy'].append(copy.deepcopy(r['policy'][0]))):
            rows = self.rows()
            mutation(rows[2])
            with self.assertRaises(capture.CaptureError):
                capture.inspect(self.save('invalid', rows))

    def test_report_escapes_html_and_preserves_existing_files(self):
        path = self.root / 'report.html'
        capture.write_reports({'text': '<script>bad()</script>'}, html_path=path)
        self.assertNotIn('<script>', path.read_text())
        with self.assertRaises(FileExistsError):
            capture.write_reports({}, html_path=path)

    def test_empty_capture_and_invalid_tolerance(self):
        with self.assertRaisesRegex(capture.CaptureError, 'no complete callbacks'):
            capture.inspect(self.save('empty', self.rows()[:2]), True)
        ref = self.save('ref')
        with self.assertRaisesRegex(capture.CaptureError, 'tolerances'):
            capture.compare(ref, [ref], obs_atol=float('nan'))

if __name__ == '__main__':
    unittest.main()
