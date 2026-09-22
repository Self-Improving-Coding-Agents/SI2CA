"""Offline model resolution tests: no model downloads, GPU imports or server processes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from huggingface_hub.errors import LocalEntryNotFoundError
from si2ca.cli import build_parser
from si2ca.model_files import complete_files, parser_defaults, resolve_files
from si2ca.run import resolve
from si2ca.serve import launch, main as serve_main
from si2ca.serving import matching_server

NUM_GPUS = 0


class ModelFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text('{"model_type":"qwen3_5_moe"}')

    def weights(self):
        (self.root / 'model.safetensors').write_bytes(b'offline fixture, not real weights')

    def test_explicit_local_model_never_calls_hub(self):
        self.weights()
        with patch('si2ca.model_files.snapshot_download') as download:
            self.assertEqual(resolve_files(str(self.root)), str(self.root))
            self.assertEqual(resolve_files('Qwen/model', local_path=str(self.root)), str(self.root))
        download.assert_not_called()

    def test_complete_cache_is_offline_first(self):
        self.weights()
        with patch('si2ca.model_files.snapshot_download', return_value=str(self.root)) as download:
            resolve_files('Qwen/model', revision='commit', cache_dir='/cache/models')
        download.assert_called_once_with(repo_id='Qwen/model', revision='commit', cache_dir='/cache/models', local_files_only=True)

    def test_partial_sharded_cache_downloads_missing_files(self):
        names = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': dict(zip(['a','b'], names))}))
        (self.root / names[0]).write_bytes(b'shard 1')
        self.assertFalse(complete_files(self.root))
        def download(**kwargs):
            if not kwargs.get('local_files_only'):
                (self.root / names[1]).write_bytes(b'shard 2')
            return str(self.root)
        with patch('si2ca.model_files.snapshot_download', side_effect=download) as mocked:
            self.assertEqual(resolve_files('Qwen/model'), str(self.root))
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(complete_files(self.root))

    def test_cache_miss_downloads(self):
        self.weights()
        with patch('si2ca.model_files.snapshot_download', side_effect=[LocalEntryNotFoundError('not cached'), str(self.root)]) as download:
            resolve_files('Qwen/model', revision='v1')
        self.assertEqual(download.call_args.kwargs['revision'], 'v1')
        self.assertIsNone(download.call_args.kwargs['allow_patterns'])

    def test_training_metadata_cannot_hide_a_missing_weight_shard(self):
        (self.root / 'training_args.bin').write_bytes(b'training arguments, not weights')
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
            {'a':'model-00001-of-00001.safetensors', 'b':'additional.safetensors'}}))
        (self.root / 'model-00001-of-00001.safetensors').write_bytes(b'present but incomplete indexed set')
        self.assertFalse(complete_files(self.root))

    def test_offline_partial_cache_fails_without_network_retry(self):
        with patch('si2ca.model_files.snapshot_download', return_value=str(self.root)) as download:
            with self.assertRaisesRegex(ValueError, 'forbids downloading'):
                resolve_files('Qwen/model', local_files_only=True)
        self.assertEqual(download.call_count, 1)

    def test_tokenizer_download_excludes_weights(self):
        def download(**kwargs):
            if not kwargs.get('local_files_only'):
                (self.root / 'tokenizer.json').write_text('{}')
                self.assertNotIn('*.safetensors', kwargs['allow_patterns'])
                self.assertNotIn('*.bin', kwargs['allow_patterns'])
            return str(self.root)
        with patch('si2ca.model_files.snapshot_download', side_effect=download):
            resolve_files('Qwen/model', tokenizer_only=True)
        self.assertFalse(complete_files(self.root))

    def test_custom_download_directory(self):
        self.weights()
        target = self.root / 'missing'
        with patch('si2ca.model_files.snapshot_download', return_value=str(self.root)) as download:
            resolve_files('Qwen/model', local_path=str(target))
        self.assertEqual(download.call_args.kwargs['local_dir'], str(target))

    def test_lfs_pointer_and_incomplete_unindexed_shards(self):
        (self.root / 'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:missing')
        self.assertFalse(complete_files(self.root))
        (self.root / 'model-00001-of-00002.safetensors').write_bytes(b'1')
        self.assertFalse(complete_files(self.root))
        (self.root / 'model-00002-of-00002.safetensors').write_bytes(b'2')
        self.assertTrue(complete_files(self.root))

    def test_full_model_id_survives_and_sl_tokenizer_is_inferred(self):
        for extra in ([], ['--base-url','http://localhost:8151']):
            with patch('si2ca.model_files.snapshot_download') as download:
                c = resolve(build_parser().parse_args(['run','--method','SL','--model','Qwen/Qwen3.5-122B-A10B',*extra]))
            self.assertEqual(c['model'], 'Qwen/Qwen3.5-122B-A10B')
            self.assertEqual(c['tokenizer_path'], c['model'])
            download.assert_not_called()

    def test_no_qwen_parser_for_unrecognized_architecture(self):
        self.assertEqual(parser_defaults('org/any-supported-model'), (None, None))
        self.assertEqual(parser_defaults('Qwen/Qwen3.5-122B-A10B'), ('qwen3_coder','qwen3'))
        self.assertEqual(parser_defaults('meta-llama/Llama-3.3-70B-Instruct'), ('llama3',None))

    def test_generic_server_launches_with_explicit_parsers(self):
        args = SimpleNamespace(model_path='org/model',name=None,gpus='0',tp=1,port=8151,
            context_length=None,mem_fraction=.8,tool_call_parser='custom',reasoning_parser='none',
            log=str(self.root / 'serve.log'),rocm=False)
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch('si2ca.serve.resolve_files', return_value=str(self.root)) as download, \
             patch('si2ca.serve.subprocess.Popen', return_value=process) as popen, \
             patch('si2ca.serve.urllib.request.urlopen', return_value=contextlib.nullcontext(Mock(status=200))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(launch(args), process)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index('--served-model-name')+1], 'org/model')
        self.assertEqual(argv[argv.index('--tool-call-parser')+1], 'custom')
        self.assertNotIn('--reasoning-parser', argv)
        self.assertNotIn('--context-length', argv)
        download.assert_called_once_with('org/model', local_path=None, revision=None,
                                        cache_dir=None, local_files_only=False)
        popen.assert_called_once()

    def test_serving_command_launches_without_an_execution_switch(self):
        argv = ['si2ca.serve', '--model-path', 'org/model', '--gpus', '0', '--tp', '1',
                '--log', str(self.root / 'serve.log')]
        with patch('sys.argv', argv), patch('si2ca.serve.launch') as start:
            serve_main()
        start.assert_called_once()
        self.assertFalse(hasattr(start.call_args.args[0], 'execute'))
        for flag in ('--execute', '--dry-run'):
            with patch('sys.argv', argv + [flag]), patch('si2ca.serve.launch') as start, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                serve_main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()

    def test_existing_server_resolved_snapshot_matches_full_id(self):
        import httpx
        def get(url, **kwargs):
            body = {'data':[{'id':'Qwen/model'}]} if url.endswith('/models') else {'model_path':str(self.root)}
            return httpx.Response(200, json=body, request=httpx.Request('GET',url))
        with patch('si2ca.serving.httpx.get',side_effect=get), patch('huggingface_hub.snapshot_download',return_value=str(self.root)) as download:
            self.assertTrue(matching_server('http://mock','Qwen/model','Qwen/model',revision='commit'))
        self.assertTrue(download.call_args.kwargs['local_files_only'])

    def test_server_starts_without_node_scan_and_still_waits_for_readiness(self):
        args = SimpleNamespace(model_path='Qwen/model', name='Qwen/model', gpus='2,3', tp=2,
            port=8151, context_length=None, mem_fraction=.8,
            log=str(self.root / 'serve.log'), rocm=False)
        process = Mock(pid=123)
        process.poll.return_value = None
        response = Mock(status=200)
        with patch('si2ca.serve.resolve_files', return_value=str(self.root)), \
             patch('si2ca.serve.subprocess.Popen', return_value=process) as popen, \
             patch('si2ca.serve.urllib.request.urlopen', return_value=contextlib.nullcontext(response)) as health, \
             patch('subprocess.run', side_effect=AssertionError('Unexpected node inspection')), \
             patch('si2ca.serve.stop') as stop, contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(launch(args), process)
        health.assert_called_once_with('http://127.0.0.1:8151/health', timeout=5)
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2,3')
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
