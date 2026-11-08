"""Exercise the resource build rule in an isolated project under the OS temp directory."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

DEBUGGER = Path(__file__).resolve().parents[1]


class VisualizerResourceTests(unittest.TestCase):
    def test_resource_add_change_delete_missing_output_and_noop_builds(self):
        cmake = shutil.which('cmake')
        ninja = shutil.which('ninja') or os.getenv('NINJA_EXECUTABLE')
        if not cmake or not ninja:
            self.skipTest('CMake and Ninja required (or set NINJA_EXECUTABLE)')
        with tempfile.TemporaryDirectory(prefix='visualizer-sync-') as folder:
            root = Path(folder)
            source, build = root / 'source', root / 'build'
            resources = source / 'RocketSimVis'
            resources.mkdir(parents=True)
            (resources / 'first.txt').write_text('first')
            (resources / '__pycache__').mkdir()
            (resources / '__pycache__/ignored.pyc').write_bytes(b'cache')
            rules = (DEBUGGER / 'CMakeLists.txt').read_text().split('# Track individual resources', 1)[1]
            (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.12)\nproject(ResourceSync NONE)\nadd_custom_target(ForgeDebugger)\n# Track individual resources' + rules)
            shutil.copyfile(DEBUGGER / 'SyncVisualizer.cmake', source / 'SyncVisualizer.cmake')
            def run(*args, success=True):
                result = subprocess.run([cmake, *map(str, args)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                return result
            run('-S', source, '-B', build, '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + ninja)
            def update():
                return run('--build', build, '--target', 'CopyRocketSimVis')
            output = build / 'ForgeDebugger/RocketSimVis'
            stamp = build / 'visualizer-resources.stamp'
            update()
            self.assertEqual((output / 'first.txt').read_text(), 'first')
            self.assertFalse((output / '__pycache__').exists())
            stamp_time = stamp.stat().st_mtime_ns
            first_time = (output / 'first.txt').stat().st_mtime_ns
            update()
            self.assertEqual(stamp.stat().st_mtime_ns, stamp_time)
            self.assertEqual((output / 'first.txt').stat().st_mtime_ns, first_time)
            (resources / 'first.txt').write_text('updated')
            update()
            self.assertEqual((output / 'first.txt').read_text(), 'updated')
            (resources / 'second.txt').write_text('second')
            update()
            self.assertEqual((output / 'second.txt').read_text(), 'second')
            (output / 'local.txt').write_text('user-owned')
            (resources / 'first.txt').unlink()
            update()
            self.assertFalse((output / 'first.txt').exists())
            self.assertEqual((output / 'local.txt').read_text(), 'user-owned')
            (output / 'second.txt').unlink()
            update()
            self.assertEqual((output / 'second.txt').read_text(), 'second')
            # Reject parent traversal in either manifest before any removal/copy.
            outside = output.parent / 'outside.txt'
            outside.write_text('keep')
            manifest = build / 'visualizer-resources.txt'
            previous = build / 'visualizer-resources-previous.txt'
            for unsafe in ('../outside.txt', '..\\outside.txt'):
                for target in (manifest, previous):
                    with self.subTest(path=unsafe, manifest=target.name):
                        manifest.write_text('second.txt\n')
                        previous.write_text('second.txt\n')
                        target.write_text(unsafe + '\n')
                        run('-DSOURCE=' + str(resources), '-DDESTINATION=' + str(output),
                            '-DMANIFEST=' + str(manifest), '-DPREVIOUS_MANIFEST=' + str(previous),
                            '-DSTAMP=' + str(stamp), '-P', source / 'SyncVisualizer.cmake', success=False)
                        self.assertEqual(outside.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
