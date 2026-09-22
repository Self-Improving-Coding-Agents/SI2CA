"""CLI/library contracts, backend capabilities and managed-serving ownership."""
import asyncio
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock, AsyncMock

import httpx
import yaml

from si2ca.api import run_experiment
from si2ca.backends import chat_payload, check_loglikelihood
from si2ca.cli import build_parser, main
from si2ca.data import RESOURCE_ROOT
from si2ca.gold import extract_patch
from si2ca.run import resolve, execute_config, run_strategy, parser as run_parser, main as run_main
from si2ca.serving import managed_service, matching_server
from si2ca.strategies.discovered import Strategy
from si2ca.runtime.strategy import SelfGuideRubrics, TurnContext

PATCH = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b'
NUM_GPUS = 0


def configuration(*options):
    return resolve(build_parser().parse_args(['run', '--method', 'SJ', '--backend', 'api',
        '--model', 'test-model', '--base-url', 'http://localhost:9999', *options]))


class PublicCLI(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('si2ca.runtime.sandbox.create_sandbox',
                               side_effect=AssertionError('Offline CLI tests must never create a sandbox')))

    def test_strategy_handoff_uses_explicit_scaffold_names(self):
        cases = [('Standard', [], 'none'), ('SJ', [], 'SJ'),
                 ('SJ', ['--no-gold-patch'], 'none'), ('SL', [], 'SL'),
                 ('Discovered', [], 'SJ'), ('SFT', [], 'none')]
        for method, options, expected in cases:
            with self.subTest(method=method, options=options), tempfile.TemporaryDirectory() as tmp:
                cfg = configuration('--method', method, '--backend', 'self-hosted',
                                    '--benchmark', 'validation', '--tokenizer-path', tmp,
                                    '--out', tmp, *options)
                self.assertEqual(cfg['engine'], 'strategy')
                rows = [{'metadata': {'instance_id': 'example', 'gold_patch': PATCH}}]
                with patch('si2ca.runtime.strategy.amain', new_callable=AsyncMock) as run:
                    asyncio.run(run_strategy(cfg, rows))
                run.assert_awaited_once()
                self.assertEqual(run.await_args.args[0].scaffold, expected)

    def test_benchmark_listing_uses_recorded_validation_flags(self):
        output = io.StringIO()
        rows = [{'metadata': {'gold_validated': True}},
                {'metadata': {'gold_validated': False}}, {'metadata': {}}]
        with patch('si2ca.cli.read_rows', return_value=rows), \
             patch('si2ca.cli.unpack_assets', side_effect=AssertionError('Unexpected extraction')), \
             contextlib.redirect_stdout(output):
            main(['ls', 'benchmarks'])
        self.assertIn('validation 3 (SWE-bench Multilingual; 1 gold-validated)', output.getvalue())
        self.assertIn('verified 500\npro 731\nboth 1231\n', output.getvalue())

    def test_all_public_methods_dispatch_the_full_pro_split(self):
        for benchmark, count in (('pro', 731), ('both', 1231)):
            for method in ('Standard', 'SJ', 'SL', 'Discovered', 'SFT'):
                with self.subTest(method=method, benchmark=benchmark), tempfile.TemporaryDirectory() as tmp:
                    cfg = configuration('--method', method, '--backend', 'self-hosted',
                                        '--benchmark', benchmark, '--tokenizer-path', tmp, '--out', tmp)
                    self.assertEqual(cfg['limit'], 0)
                    with patch('si2ca.run.environment'), \
                         patch('si2ca.serving.managed_service', return_value=contextlib.nullcontext()), \
                         patch('si2ca.model_files.resolve_files', return_value=tmp), \
                         patch('si2ca.backends.check_loglikelihood'), \
                         patch('si2ca.run.run_strategy', new_callable=AsyncMock) as strategy, \
                         contextlib.redirect_stdout(io.StringIO()):
                        execute_config(cfg)
                    strategy.assert_awaited_once()
                    rows = strategy.await_args.args[1]
                    self.assertEqual(len(rows), count)
                    pro = [r for r in rows if r['metadata'].get('swepro')]
                    self.assertEqual(len(pro), 731)
                    self.assertEqual({r['metadata']['language'] for r in pro}, {'python', 'go', 'js', 'ts'})

    def test_deepswe_dispatches_all_tasks_with_bundled_gold_for_sj(self):
        for method in ('Standard', 'SJ'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                cfg = configuration('--method', method, '--benchmark', 'deepswe', '--out', tmp)
                self.assertEqual(cfg['engine'], 'strategy')
                with patch('si2ca.run.environment'), \
                     patch('si2ca.serving.managed_service', return_value=contextlib.nullcontext()), \
                     patch('si2ca.run.run_strategy', new_callable=AsyncMock) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    execute_config(cfg)
                run.assert_awaited_once()
                rows = run.await_args.args[1]
                self.assertEqual(len(rows), 113)
                if method == 'SJ':
                    self.assertTrue(all(r['metadata'].get('gold_patch') for r in rows))

    def test_legacy_harness_overrides_are_not_silently_accepted(self):
        for key, value, message in [('engine', 'branch', 'engine: strategy'),
                                    ('rubric_family', True, 'general rubric'),
                                    ('judge_batch_samples', True, 'independent requests')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                registry = yaml.safe_load((RESOURCE_ROOT/'configs/experiments.yaml').read_text())
                registry['settings']['sj'][key] = value
                config = Path(tmp)/'experiments.yaml'
                config.write_text(yaml.safe_dump(registry))
                with self.assertRaisesRegex(ValueError, message):
                    configuration('--backend', 'self-hosted', '--config', str(config))

    def test_search_skills_are_discoverable_and_readable_offline(self):
        output = io.StringIO()
        with patch('si2ca.cli.load_dataset', side_effect=AssertionError('Unexpected data loading')), \
             patch('si2ca.cli.unpack_assets', side_effect=AssertionError('Unexpected asset extraction')), \
             patch('subprocess.run', side_effect=AssertionError('Unexpected process')), \
             contextlib.redirect_stdout(output):
            main(['ls', 'skills'])
        found = {}
        for line in output.getvalue().splitlines():
            name, location = line.split(': ', 1)
            path = Path(location)
            self.assertTrue(path.is_absolute())
            self.assertEqual(path, RESOURCE_ROOT / 'skills' / (name + '.md'))
            content = path.read_text()
            self.assertTrue(content.startswith('---\n'))
            metadata = yaml.safe_load(content.split('---', 2)[1])
            self.assertEqual(metadata['name'], name.lower())
            self.assertTrue(metadata['description'].strip())
            found[name] = path
        self.assertEqual(set(found), {'Proposal', 'Recorder'})
        self.assertEqual(set((RESOURCE_ROOT / 'skills').iterdir()), set(found.values()))

    def test_proposer_example_matches_the_live_strategy_contract(self):
        skill = RESOURCE_ROOT / 'skills/Proposal.md'
        content = skill.read_text()
        example = content.split('```python\n', 1)[1].split('```', 1)[0]
        namespace = {}
        exec(compile(example, str(skill), 'exec'), namespace)
        strategy = namespace['Strategy']()
        for returncode, expected in ((None, 1), (0, 1), (1, 2)):
            ctx = TurnContext(1, [], 'synthetic observation', returncode, False, ['ls'])
            self.assertEqual(strategy.plan(ctx), expected)
        judge = AsyncMock(side_effect=AssertionError('Single candidate needs no judge'))
        selected, label, usage = asyncio.run(strategy.select(ctx, [{'command': 'ls'}], judge))
        self.assertEqual((selected, label, usage), (0, 'only_valid_toolcall', {}))
        judge.assert_not_awaited()
        declaration = json.loads(content.split('```json\n', 1)[1].split('```', 1)[0])
        self.assertEqual(len(declaration['candidates']), 1)
        self.assertEqual(declaration['candidates'][0]['scaffold'], 'SJ')

    def test_resource_preflight_is_not_exposed(self):
        import importlib.util
        self.assertNotIn('preflight', build_parser().format_help())
        self.assertIsNone(importlib.util.find_spec('si2ca.preflight'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            main(['preflight'])
        self.assertEqual(cm.exception.code, 2)

    def test_execution_does_not_scan_node_or_write_preflight_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = configuration('--no-gold-patch', '--out', tmp)
            cfg['public_cli'] = True
            rows = [{'metadata': {'instance_id': 'mock-task'}}]
            with patch('si2ca.run.load_dataset', return_value=rows), \
                 patch('si2ca.run.environment'), \
                 patch('si2ca.serving.managed_service', return_value=contextlib.nullcontext()), \
                 patch('si2ca.run.run_strategy', new_callable=AsyncMock) as evaluate, \
                 patch('subprocess.run', side_effect=AssertionError('Unexpected node inspection')), \
                 contextlib.redirect_stdout(io.StringIO()):
                execute_config(cfg)
            evaluate.assert_awaited_once_with(cfg, rows)
            self.assertTrue((Path(tmp) / 'config.json').is_file())
            self.assertFalse((Path(tmp) / 'preflight.json').exists())

    def test_help_and_listing_are_offline(self):
        for argv in (['--help'], ['run','-h'], ['serve','--help'], ['--version']):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main(argv)
            self.assertEqual(cm.exception.code, 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output): main(['ls','methods'])
        self.assertIn('Discovered Early-Commit Window Strategy', output.getvalue())
        self.assertNotIn('Strategy6', output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            main(['run', '--help'])
        self.assertEqual(cm.exception.code, 0)
        help_text = ' '.join(output.getvalue().split())
        self.assertIn('Discovered Early-Commit Window Strategy', help_text)
        self.assertNotIn('strategy6', help_text.lower())

    def test_public_and_module_commands_dispatch_without_execution_switches(self):
        with patch('si2ca.cli.execute_config') as execute:
            main(['run', '--method', 'SJ', '--model', 'policy',
                  '--base-url', 'http://localhost:9999', '--out', 'runs/test'])
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0]['setting'], 'sj')
        self.assertNotIn('execute', execute.call_args.args[0])
        with patch('sys.argv', ['si2ca.run', '--setting', 'sj', '--model', 'policy',
                  '--base-url', 'http://localhost:9999', '--out', 'runs/test']), \
             patch('si2ca.run.execute_config') as execute:
            run_main()
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0]['setting'], 'sj')
        self.assertNotIn('execute', execute.call_args.args[0])

    def test_removed_execution_switches_are_rejected_before_running(self):
        for flag in ('--dry-run', '--execute'):
            with self.subTest(flag=flag), patch('si2ca.cli.execute_config') as execute, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['run', '--method', 'SJ', '--model', 'policy', flag])
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run_parser().parse_args(['--setting', 'sj', '--model', 'policy',
                    '--base-url', 'http://localhost:9999', '--out', 'runs/test', flag])
            self.assertEqual(error.exception.code, 2)

    def test_library_rejects_unknown_keywords_even_when_false_or_none(self):
        for key in ('dry_run', 'execute', 'unknown_option'):
            for value in (True, False, None):
                with self.subTest(key=key, value=value), patch('si2ca.api.subprocess.run') as run:
                    with self.assertRaisesRegex(TypeError, 'Unknown experiment option'):
                        run_experiment(method='SJ', model='Qwen/model', **{key: value})
                    run.assert_not_called()

    def test_unvalidated_tasks_still_fail_before_serving(self):
        cfg = configuration('--benchmark', 'validation', '--no-gold-patch')
        rows = [{'metadata': {'instance_id': 'pending-task', 'gold_validated': False}}]
        with patch('si2ca.run.load_dataset', return_value=rows), \
             patch('si2ca.serving.managed_service') as service, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Validate the remaining grading chains'):
                execute_config(cfg)
        service.assert_not_called()

    def test_sj_knobs_and_weight_normalization(self):
        cfg = configuration('--branch','4','--average','5','--gold-weight','.45')
        self.assertEqual((cfg['k'],cfg['score_samples']), (4,5))
        selector = SelfGuideRubrics()
        selector.GOLD_WEIGHT = cfg['gold_weight']
        weights = selector._weights(cfg['pi'] == 'gold')
        self.assertEqual(weights['G1'], .45)
        self.assertAlmostEqual(weights['R1'], .15*.55)
        self.assertAlmostEqual(sum(weights.values()), 1.)

    def test_no_gold_removes_g1(self):
        cfg = configuration('--no-gold-patch')
        self.assertEqual(cfg['setting'], 'sj_no_pi')
        self.assertNotIn('G1',SelfGuideRubrics()._weights(cfg['pi'] == 'gold'))
        self.assertEqual(cfg['gold_weight'], 0.)

    def test_conflicting_and_invalid_gold_parameters(self):
        for options in [('--pi','gold','--no-gold-patch'), ('--gold-weight','1.1'),
                        ('--gold-weight','nan'), ('--no-gold-patch','--gold-weight','.4')]:
            with self.assertRaises(ValueError): configuration(*options)

    def test_sl_rejected_for_api_before_tokenizer_loading(self):
        with self.assertRaisesRegex(ValueError,'SL requires --backend self-hosted'):
            configuration('--method','SL')

    def test_sl_matrix_resolves_from_method_options(self):
        cfg = configuration('--method','SL','--backend','self-hosted','--tokenizer-path','Qwen/model',
                            '--pi','gold','--placement','tail','--score','gain')
        self.assertEqual(cfg['setting'],'sl_gold_tail_gain')

    def test_auto_serving_infers_endpoint_tokenizer_and_tp(self):
        args=build_parser().parse_args(['run','--method','SL','--model','Qwen/model',
            '--gpu-ids','2,3'])
        cfg=resolve(args)
        self.assertEqual(cfg['managed_model_path'],'Qwen/model')
        self.assertEqual(cfg['tokenizer_path'],'Qwen/model')
        self.assertEqual(cfg['base_url'],'http://127.0.0.1:8151')
        self.assertEqual(cfg['model'],'Qwen/model')

    def test_discovered_early_commit_threshold(self):
        cfg=configuration('--method','Discovered','--branch-until-edit','5')
        self.assertEqual((cfg['strategy'],cfg['branch_until_edit']),('discovered',5))
        for alias in ('Discovered', 'discovered', 'DISCOVERED', 'Strategy6', 'strategy6', 'STRATEGY6'):
            with self.subTest(method=alias):
                self.assertEqual(configuration('--method',alias,'--branch-until-edit','5','--out','runs/test'),
                                 configuration('--method','Discovered','--branch-until-edit','5','--out','runs/test'))
        for n in (1,3,5):
            with patch.object(Strategy,'MAX_PRIOR_MUTATIONS',n-1):
                for edits in range(6):
                    ctx=TurnContext(0,[],'',0,True,['ls']+['sed -i x a']*edits)
                    self.assertEqual(Strategy().plan(ctx),2 if edits<n else 1)

    def test_missing_gold_prevents_all_serving(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'tasks.jsonl'
            file.write_text(json.dumps({'prompt':'fix','metadata':{'instance_id':'missing-patch',
                'image':'never-start','problem_statement':'fix','eval_cmd':'true'}})+'\n')
            cfg=configuration('--dataset',str(file),'--out',str(Path(tmp)/'run'))
            with patch('si2ca.serving.managed_service') as service:
                with self.assertRaisesRegex(ValueError,'missing-patch'): execute_config(cfg)
                service.assert_not_called()
            self.assertFalse((Path(tmp)/'run').exists())

    def test_library_uses_isolated_interpreter(self):
        with patch('si2ca.api.subprocess.run') as run:
            run_experiment(method='SJ',model='Qwen/model',gpu_ids='0,1',judge_samples=5,gold_patch=False)
        argv=run.call_args.args[0]
        self.assertIn('--method',argv)
        self.assertIn('--gpu-ids',argv)
        self.assertIn('--judge-samples',argv)
        self.assertIn('--no-gold-patch',argv)
        self.assertTrue(run.call_args.kwargs['check'])

    def test_no_pi_does_not_require_disambiguating_unused_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'tasks.jsonl'
            file.write_text(json.dumps({'prompt':'fix','patch':PATCH,
                'metadata':{'instance_id':'no-pi','image':'unused','problem_statement':'fix',
                            'gold_patch':PATCH+'\n+c','eval_cmd':'true'}})+'\n')
            cfg=configuration('--no-gold-patch','--dataset',str(file),'--out',str(Path(tmp)/'run'))
            with patch('si2ca.run.environment'), \
                 patch('si2ca.serving.managed_service', return_value=contextlib.nullcontext()), \
                 patch('si2ca.run.run_strategy', new_callable=AsyncMock) as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                execute_config(cfg)
            evaluate.assert_awaited_once()


class GoldAndBackends(unittest.TestCase):
    def test_sl_capability_requires_actual_native_logprobs(self):
        request=httpx.Request('POST','http://mock/generate')
        good=httpx.Response(200,request=request,json={'meta_info':{'input_token_logprobs':[[None,1],[-.4,2]]}})
        with patch('si2ca.backends.httpx.post',return_value=good) as post:
            check_loglikelihood('http://mock')
            self.assertEqual(post.call_args.kwargs['json']['sampling_params']['max_new_tokens'],0)
        bad=httpx.Response(200,request=request,json={'meta_info':{}})
        with patch('si2ca.backends.httpx.post',return_value=bad):
            with self.assertRaisesRegex(ValueError,'cannot run'):check_loglikelihood('http://mock')

    def test_patch_aliases_and_diff_fences(self):
        for row in ({'patch':PATCH},{'reference_patch':PATCH},{'golden_patch':PATCH},
                    {'metadata':{'answer':f'Here is a fix:\n```diff\n{PATCH}\n```'}},
                    {'solution':{'patch':PATCH}}):
            self.assertEqual(extract_patch(row),PATCH+'\n')

    def test_extracted_patch_is_git_parseable_and_keeps_context_whitespace(self):
        diff = ('diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n'
                '@@ -1,2 +1,2 @@\n-old\n+new\n \n')
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git', 'init', '--quiet', tmp], check=True)
            for original, expected in [(diff, diff), (diff[:-1], diff)]:
                Path(tmp, 'a.txt').write_text('old\n\n')
                actual = extract_patch({'patch': original})
                self.assertEqual(actual, expected)
                subprocess.run(['git', 'apply', '-'], input=actual, text=True,
                               cwd=tmp, check=True, capture_output=True)
                self.assertEqual(Path(tmp, 'a.txt').read_text(), 'new\n\n')

    def test_prose_answers_are_not_patches(self):
        self.assertIsNone(extract_patch({'answer':'Change the parser'}))
        self.assertIsNone(extract_patch({'test_patch':PATCH}))

    def test_conflicting_patches_require_explicit_field(self):
        row={'patch':PATCH,'metadata':{'reference_patch':PATCH+'\n+c'}}
        with self.assertRaisesRegex(ValueError,'conflicting'):extract_patch(row)
        self.assertEqual(extract_patch(row,'metadata.reference_patch'),PATCH+'\n+c\n')

    def test_api_request_omits_local_only_fields(self):
        payload={'model':'x','messages':[{'role':'user','content':'task'}],
            'top_k':10,'temperature':1.,'top_p':.95,'logprobs':True,
            'chat_template_kwargs':{'enable_thinking':True},'max_tokens':4096}
        with patch.dict(os.environ,{'SI2CA_BACKEND':'api','SI2CA_API_TOKEN_FIELD':'max_completion_tokens','SI2CA_API_SAMPLING':'0'}):
            body=chat_payload(payload)
        self.assertEqual(body['max_completion_tokens'],4096)
        self.assertNotIn('top_k',body);self.assertNotIn('temperature',body)
        self.assertNotIn('chat_template_kwargs',body)
        self.assertEqual(body['messages'],payload['messages'])
        self.assertIn('top_k',payload)

    def test_self_hosted_request_is_unchanged(self):
        payload={'top_k':10,'logprobs':True}
        with patch.dict(os.environ,{'SI2CA_BACKEND':'self-hosted'}):
            self.assertIs(chat_payload(payload),payload)


class ServingOwnership(unittest.TestCase):
    def config(self):
        return dict(managed_model_path='Qwen/model',model='model',base_url='http://localhost:8151',
            gpu_ids='0,1',tp=None,port=8151,context_length=262144,mem_fraction=.8,out='/tmp/mock',rocm=False)

    def test_reused_server_is_never_stopped(self):
        with patch('si2ca.serving.matching_server',return_value=True), patch('si2ca.serve.launch') as launch, patch('si2ca.serve.stop') as stop:
            with managed_service(self.config()):pass
        launch.assert_not_called();stop.assert_not_called()

    def test_owned_server_is_stopped_after_failure(self):
        process=SimpleNamespace(pid=123)
        with patch('si2ca.serving.matching_server',return_value=False), \
             patch('importlib.util.find_spec',return_value=object()), \
             patch('si2ca.serve.launch',return_value=process), patch('si2ca.serve.stop') as stop:
            with self.assertRaises(RuntimeError):
                with managed_service(self.config()):raise RuntimeError('run failed')
            stop.assert_called_once_with(process)

    def test_keep_server_is_explicit(self):
        with patch('si2ca.serving.matching_server',return_value=False), patch('importlib.util.find_spec',return_value=object()), \
             patch('si2ca.serve.launch',return_value=SimpleNamespace(pid=123)), patch('si2ca.serve.stop') as stop:
            with managed_service(self.config() | {'keep_server':True}):pass
            stop.assert_not_called()

    def test_resume_preserves_previous_server_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'serve.log';old.write_text('previous run')
            with patch('si2ca.serving.matching_server',return_value=False), patch('importlib.util.find_spec',return_value=object()), \
                 patch('si2ca.serve.launch',return_value=SimpleNamespace(pid=123)) as launch, patch('si2ca.serve.stop'):
                with managed_service(self.config() | {'out':tmp}):pass
                self.assertNotEqual(launch.call_args.args[0].log,str(old))
                self.assertEqual(old.read_text(),'previous run')

    def test_wrong_existing_model_is_rejected(self):
        response=httpx.Response(200,json={'data':[{'id':'other'}]},request=httpx.Request('GET','http://mock'))
        with patch('si2ca.serving.httpx.get',return_value=response):
            with self.assertRaisesRegex(ValueError,'already serves'):matching_server('http://mock','model','Qwen/model')


if __name__=='__main__':unittest.main()
