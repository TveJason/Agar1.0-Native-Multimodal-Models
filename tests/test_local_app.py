import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from local_app import Controller, load_settings

class Runtime:
    def __init__(self, root):
        self.cfg_module = SimpleNamespace()
        self._current_session_dir = root
    def reset_stop(self): pass
    def update_sampling_config(self, mode): self.mode = mode
    def encode_and_set_prompt(self, sample): self.sample = sample
    def stream_events(self):
        yield {'type': 'text', 'text': 'Hello'}
        yield {'type': 'image', 'paths': [str(Path(self._current_session_dir) / 'image.png')]}
        yield {'type': 'text', 'text': 'After image'}

class Tests(unittest.TestCase):
    def test_stream_order_and_save(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root)
            controller = Controller({'output_dir': root}, runtime)
            frames = list(controller.generate('Explain', None, 'Interleaved text + images'))
            result, files, status = frames[-1]
            self.assertEqual(runtime.mode, 'howto')
            self.assertEqual([m['content'] for m in result], ['Hello', {'path': str(Path(root)/'image.png')}, 'After image'])
            self.assertTrue((Path(root)/'response.json').exists())
            self.assertFalse(controller.lock.locked())
    def test_reference_switch_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root)
            controller = Controller({'output_dir': root}, runtime)
            list(controller.generate('Edit', 'input.png', 'Text to image'))
            self.assertEqual(runtime.mode, 'x2i')
            self.assertEqual(runtime.sample['images'], ['input.png'])
            with self.assertRaises(ValueError): list(controller.generate(' ', None, 'Text to image'))
    def test_failure_unlocks(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root)
            def fail(): raise RuntimeError('test failure')
            runtime.stream_events = fail
            controller = Controller({'output_dir': root}, runtime)
            with self.assertRaises(RuntimeError): list(controller.generate('Test', None, 'Text to image'))
            self.assertFalse(controller.lock.locked())

if __name__ == '__main__': unittest.main()
