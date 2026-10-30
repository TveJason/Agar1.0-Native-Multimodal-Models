"""Shared local paths for the browser app and existing CLI entry points.

This file imports the upstream task defaults, then overrides machine settings.
"""
import json
import os
from pathlib import Path
from configs.example_config_visual_guidance import *

_settings = json.loads(Path(os.environ.get('AGAR_CONFIG', 'local_config.json')).read_text())
model_path = _settings['model_path']
tokenizer_path = _settings['tokenizer_path']
vq_path = _settings['vq_path']
save_path = _settings['output_dir']
hf_device = _settings.get('hf_device', 'auto')
vq_device = _settings.get('vq_device', 'cuda:0')
