"""Agar local UI; reuses BAAI's streaming model runtime."""
import argparse
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    for key in ('model_path', 'tokenizer_path', 'vq_path', 'output_dir'):
        if not settings.get(key):
            raise ValueError(f'Missing configuration: {key}')
    return settings


class Controller:
    def __init__(self, settings, runtime=None):
        self.settings = settings
        self.runtime = runtime
        self.lock = threading.Lock()

    def generate(self, prompt, image, mode):
        if not prompt.strip():
            raise ValueError('Enter a prompt first.')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Another generation is running. Wait or press Stop.')
        try:
            if self.runtime is None:
                from model_runtime import ModelRuntime
                self.runtime = ModelRuntime()
                self.runtime.initialize('configs/local_app_config.py', self.settings['output_dir'])
            runtime = self.runtime
            task = 'howto' if mode == 'Interleaved text + images' else ('x2i' if image else 't2i')
            runtime.reset_stop()
            runtime.update_sampling_config(task)
            # Keep checkpoint, paths and device settings from the local config;
            # only task sampling/template settings come from upstream presets.
            runtime._save_dir = str(Path(self.settings['output_dir']).resolve())
            runtime.cfg_module.save_path = runtime._save_dir
            runtime.encode_and_set_prompt({'text': prompt, 'images': [image] if image else []})
            messages = []
            text = ''
            files = []
            status = 'Generating…'
            yield messages, files, status
            for event in runtime.stream_events():
                if event['type'] == 'text':
                    text += event['text']
                    if messages and isinstance(messages[-1]['content'], str):
                        messages[-1]['content'] = text
                    else:
                        messages.append({'role': 'assistant', 'content': text})
                elif event['type'] == 'image':
                    for path in event.get('paths', []):
                        files.append(path)
                        messages.append({'role': 'assistant', 'content': {'path': path}})
                    text = ''
                    status = 'Image saved; continuing…'
                elif event['type'] == 'broken_image':
                    status = 'Generating image tokens…'
                yield messages, list(files), status
            session = Path(runtime._current_session_dir)
            (session / 'response.json').write_text(json.dumps(messages, ensure_ascii=False, indent=2))
            files = [str(p) for p in sorted(session.iterdir()) if p.is_file()]
            yield messages, files, f'Finished. Saved in {session}'
        finally:
            self.lock.release()

    def stop(self):
        if self.runtime:
            self.runtime.request_stop()
        return 'Stop requested; waiting for the next generation event.'


def build_ui(controller):
    import gradio as gr
    with gr.Blocks(title='Agar 1.0 · Emu3.5') as app:
        gr.Markdown('# Agar 1.0\nLocal Emu3.5 studio · text to image and interleaved generation')
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(['Text to image', 'Interleaved text + images'], value='Text to image', label='Mode')
                prompt = gr.Textbox(label='Prompt', lines=6, placeholder='Describe an image, or ask for an illustrated explanation…')
                image = gr.Image(type='filepath', label='Optional reference image')
                gr.Markdown('Use the general BAAI/Emu3.5 checkpoint for both modes. Text streams; images appear when decoding finishes.')
                send = gr.Button('Generate', variant='primary')
                stop = gr.Button('Stop')
            with gr.Column(scale=2):
                result = gr.Chatbot(type='messages', label='Generated output', height=550)
                status = gr.Markdown('Ready. The model loads on the first request.')
                files = gr.Files(label='Saved files', interactive=False)
        send.click(controller.generate, [prompt, image, mode], [result, files, status], concurrency_limit=1, concurrency_id='model')
        stop.click(controller.stop, outputs=status, queue=False)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local_config.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    os.environ['AGAR_CONFIG'] = str(Path(args.config).resolve())
    settings = load_settings(args.config)
    Path(settings['output_dir']).mkdir(parents=True, exist_ok=True)
    build_ui(Controller(settings)).queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == '__main__':
    main()
