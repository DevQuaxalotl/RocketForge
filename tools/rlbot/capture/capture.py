"""Inspect and compare optional RocketForge RLBot callback recordings."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import html
import json
import math
from pathlib import Path
import sys
import zlib

SCHEMA = "rocketforge.rlbot.capture.v1"
MAX_RECORD = 64 * 1024 * 1024


class CaptureError(ValueError):
    pass


class Recording:
    """Streaming reader; a truncated last line is never a complete callback."""
    def __init__(self, path, allow_incomplete=False):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.allow_incomplete = allow_incomplete
        self.sequence = self.callbacks = 0
        self.complete = self.initialized = False
        try:
            self.header = self._read()
            if (not self.header or self.header.get("kind") != "header"
                    or self.header.get("schema") != SCHEMA):
                raise CaptureError("unsupported capture header")
            indices = self.header.get("indices")
            if (not isinstance(indices, list) or not indices or len(indices) > 64
                    or any(type(i) is not int or i < 0 or i >= 64 for i in indices)
                    or indices != sorted(set(indices))):
                raise CaptureError("invalid captured car indices")
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def _read(self):
        line = self.file.readline(MAX_RECORD + 1)
        if len(line) > MAX_RECORD:
            raise CaptureError("capture record exceeds size limit")
        if not line:
            return None
        if not line.endswith(b"\n"):
            if self.allow_incomplete and not self.complete:
                return None
            raise CaptureError("truncated capture record")
        try:
            envelope = json.loads(line)
            payload = envelope["payload"]
            if zlib.crc32(payload.encode("utf-8")) != envelope["crc32"]:
                raise CaptureError("capture checksum mismatch")
            record = json.loads(payload, parse_constant=lambda x: (_ for _ in ()).throw(CaptureError("nonfinite JSON")))
            if record["sequence"] != self.sequence:
                raise CaptureError("capture sequence gap")
            self.sequence += 1
            return record
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise CaptureError(f"invalid capture record: {error}") from error

    def __iter__(self):
        while record := self._read():
            kind = record.get("kind")
            if self.complete:
                raise CaptureError("data after capture footer")
            if kind == "initialize":
                if self.initialized or self.callbacks:
                    raise CaptureError("unexpected capture initialization")
                self.initialized = True
                self.initialization = {k: record.get(k) for k in ("team_info", "field", "match")}
            elif kind == "callback":
                if not self.initialized or record.get("callback") != self.callbacks:
                    raise CaptureError("missing initialization or callback")
                for field in ("packet", "prediction", "rng", "policy", "schedule", "controls"):
                    if field not in record:
                        raise CaptureError(f"callback is missing {field}")
                if not isinstance(record["rng"], str):
                    raise CaptureError("invalid RNG state")
                for field in ("packet", "prediction"):
                    value = record[field]
                    if value is not None and (not isinstance(value, str) or len(value) % 2
                            or any(ch not in '0123456789abcdef' for ch in value)):
                        raise CaptureError(f"invalid {field} encoding")
                for field in ("policy", "schedule", "controls"):
                    rows = record[field]
                    if not isinstance(rows, list):
                        raise CaptureError(f"invalid {field} rows")
                    ids = [row.get("index") for row in rows]
                    if len(ids) != len(set(ids)) or not set(ids) <= set(self.header["indices"]):
                        raise CaptureError(f"duplicate or unknown {field} car")
                if set(row['index'] for row in record['controls']) != set(self.header['indices']):
                    raise CaptureError("missing controlled car")
                for row in record['controls']:
                    numeric(row.get('values'), 8, 'controls')
                for row in record['policy']:
                    numeric(row.get('observation'), None, 'observation')
                    numeric(row.get('probabilities'), None, 'probabilities')
                    mask = row.get('mask')
                    if (not isinstance(mask, list) or not mask or any(v not in (0, 1) for v in mask)
                            or len(mask) != len(row['probabilities']) or not any(mask)):
                        raise CaptureError("invalid action mask")
                    action = row.get('action')
                    if type(action) is not int or not 0 <= action < len(mask) or not mask[action]:
                        raise CaptureError("invalid selected action")
                self.callbacks += 1
                yield record
            elif kind == "footer":
                if record.get("lost") != 0 or record.get("write_failed") is not False or record.get("callbacks") != self.callbacks:
                    raise CaptureError("capture reports lost data")
                self.complete = True
            else:
                raise CaptureError("unknown capture record")
        if not self.callbacks:
            raise CaptureError("capture has no complete callbacks")
        if not self.complete and not self.allow_incomplete:
            raise CaptureError("capture is incomplete; use --allow-incomplete for its verified prefix")


def numeric(values, size, label):
    if (not isinstance(values, list) or not values or (size is not None and len(values) != size)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise CaptureError(f"invalid {label} values")


class Timing:
    def __init__(self):
        self.count = 0
        self.total = self.maximum = 0.0
    def add(self, record):
        if 'inference_us' not in record:
            return
        value = record['inference_us']
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise CaptureError("invalid inference time")
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
    def report(self):
        return dict(calls=self.count, mean_us=self.total / self.count if self.count else None,
                    max_us=self.maximum if self.count else None)


def inspect(path, allow_incomplete=False):
    with Recording(path, allow_incomplete) as stream:
        timing = Timing()
        decisions = 0
        for record in stream:
            decisions += len(record['policy'])
            timing.add(record)
        return dict(path=str(path), complete=stream.complete, callbacks=stream.callbacks,
                    policy_rows=decisions, header=stream.header, inference=timing.report())


def differences(a, b, path='', tolerance=0):
    """Yield exact structure changes and finite numeric differences outside tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            child = f'{path}.{key}' if path else key
            if key not in a or key not in b:
                yield child, a.get(key), b.get(key)
            else:
                yield from differences(a[key], b[key], child, tolerance)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path + '.length', len(a), len(b)
        for i, (left, right) in enumerate(zip(a, b)):
            yield from differences(left, right, f'{path}[{i}]', tolerance)
    elif type(a) in (float, int) and type(b) in (float, int):
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance:
            yield path, a, b
    elif a != b:
        yield path, a, b


def compare(reference, candidates, allow_incomplete=False, obs_atol=1e-6, probs_atol=1e-5):
    for value in (obs_atol, probs_atol):
        if not math.isfinite(value) or value < 0:
            raise CaptureError("tolerances must be finite and nonnegative")
    examples, counts = [], {}
    compared = 0
    ref_time, cand_time = Timing(), Timing()
    def note(category, callback, car, left, right, tolerance=0):
        for path, a, b in differences(left, right, tolerance=tolerance):
            counts[category] = counts.get(category, 0) + 1
            if len(examples) < 100:
                examples.append(dict(category=category, callback=callback, car=car,
                                     field=path, reference=a, candidate=b))
    with ExitStack() as stack:
        ref = stack.enter_context(Recording(reference, allow_incomplete))
        streams = [stack.enter_context(Recording(p, allow_incomplete)) for p in candidates]
        if not streams:
            raise CaptureError("no candidate recordings")
        indices = [i for stream in streams for i in stream.header['indices']]
        if len(indices) != len(set(indices)) or set(indices) != set(ref.header['indices']):
            raise CaptureError("candidate recordings must cover each reference car exactly once")
        for stream in streams:
            for field in ('agent', 'team', 'setup', 'checkpoint'):
                if stream.header.get(field) != ref.header.get(field):
                    raise CaptureError(f"recordings differ in {field}; use the same agent and checkpoint")
        iterators = [iter(stream) for stream in streams]
        for original in ref:
            group = [next(iterator, None) for iterator in iterators]
            if any(row is None for row in group):
                raise CaptureError("candidate is missing callbacks")
            for stream, row in zip(streams, group):
                if stream.initialization != ref.initialization:
                    raise CaptureError("recorded initialization differs")
                if row['callback'] != original['callback'] or any(row[k] != original[k] for k in ('packet', 'prediction')):
                    raise CaptureError("callback inputs differ; these are not replays of the same recording")
                cand_time.add(row)
            ref_time.add(original)
            for field, tolerance in (('policy', None), ('schedule', 0), ('controls', 0)):
                expected = {row['index']: row for row in original[field]}
                actual = {row['index']: row for callback in group for row in callback[field]}
                for car in sorted(expected.keys() | actual.keys()):
                    a, b = expected.get(car), actual.get(car)
                    if field == 'policy' and a is not None and b is not None:
                        for key in a.keys() | b.keys():
                            tol = obs_atol if key == 'observation' else probs_atol if key == 'probabilities' else 0
                            note(key, compared, car, a.get(key), b.get(key), tol)
                    else:
                        note(field, compared, car, a, b, tolerance or 0)
            compared += 1
        for iterator in iterators:
            if next(iterator, None) is not None:
                raise CaptureError("candidate has extra callbacks")
        complete = ref.complete and all(stream.complete for stream in streams)
    return dict(status='different' if counts else 'match' if complete else 'prefix_match',
                complete=complete, callbacks=compared, differences=counts, examples=examples,
                reference=str(reference), candidates=[str(p) for p in candidates],
                tolerances=dict(observation=obs_atol, probabilities=probs_atol, controls=0),
                reference_inference=ref_time.report(), candidate_inference=cand_time.report())


def write_reports(report, json_path=None, html_path=None):
    text = json.dumps(report, indent=2, allow_nan=False)
    for path, content in ((json_path, text), (html_path,
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>RLBot recording comparison</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:20px}</style>'
        '<h1>RLBot recording comparison</h1><p>Controls and scheduling are compared exactly. '
        'Observation and probability tolerances are recorded below. Timings include optional diagnostics '
        'and are not a clean performance benchmark. Only the first 100 differences are shown.</p><pre>'
        + html.escape(text) + '</pre></html>')):
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Reports are deliberately created exclusively, preserving prior comparisons.
            with path.open('x', encoding='utf-8') as output:
                output.write(content + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    show = sub.add_parser('inspect')
    show.add_argument('recording', type=Path)
    diff = sub.add_parser('compare')
    diff.add_argument('reference', type=Path)
    diff.add_argument('candidates', type=Path, nargs='+', help='files or one replay output directory')
    diff.add_argument('--obs-atol', type=float, default=1e-6)
    diff.add_argument('--probs-atol', type=float, default=1e-5)
    for command in (show, diff):
        command.add_argument('--allow-incomplete', action='store_true')
        command.add_argument('--json', type=Path)
        command.add_argument('--html', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            report = inspect(args.recording, args.allow_incomplete)
        else:
            candidates = []
            for candidate in args.candidates:
                candidates.extend(sorted(candidate.rglob('*.rfcapture')) if candidate.is_dir() else [candidate])
            report = compare(args.reference, candidates, args.allow_incomplete, args.obs_atol, args.probs_atol)
        write_reports(report, args.json, args.html)
        print(json.dumps(report, indent=2, allow_nan=False))
        if report.get('status') == 'different':
            return 1
        return 0 if report['complete'] else 3
    except (CaptureError, OSError, ValueError, KeyError, TypeError) as error:
        print(f'rlbot-capture: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
