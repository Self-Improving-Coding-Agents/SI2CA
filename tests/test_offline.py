"""Offline behavioral checks. Network requests and container execution are mocked."""
NUM_GPUS = 0

import asyncio
import copy
import csv
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import yaml

from si2ca.data import ROOT, RESOURCE_ROOT, load_dataset, read_rows, unpack_assets
from si2ca.run import parser, resolve
from si2ca.runtime import strategy as runner
from si2ca.runtime.judging import render_prefix
from si2ca.strategies.discovered import Strategy as Discovered
from si2ca.strategies.sl_gain import Strategy as Gain
from si2ca.results import load_results, summary
from si2ca.search import decide


def candidate(command, nll=None, privileged=None):
    return {'command': command, 'mean_nll': nll, 'mean_nll_privileged': privileged,
            'msg': {'role':'assistant','content':'Inspect the implementation.',
                    'tool_calls':[{'id':command,'type':'function','function':{
                        'name':'bash','arguments':json.dumps({'command':command})}}]}}


def context(commands=None, gold=True):
    return runner.TurnContext(0,[{'role':'user','content':'<pr_description>Fix the bug</pr_description>'}],
                              '',0,False,commands or [],random.Random(42),gold_in_judge=gold)


class DataAndConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unpack_assets()

    def test_scaffold_cli_accepts_only_explicit_method_names(self):
        parser = runner.build_parser()
        self.assertIn('--scaffold {none,SJ,SL,SJ+SL}', parser.format_help())
        self.assertFalse(hasattr(runner, 'normalize_scaffold'))
        for name in ('none', 'SJ', 'SL', 'SJ+SL'):
            args = parser.parse_args(['--dataset', 'tasks.jsonl', '--out', 'results.json', '--scaffold', name])
            self.assertEqual(args.scaffold, name)
        # Negative fixtures only: old labels must never become aliases again.
        for invalid in ('R', 'H', 'RH', '2R', '2H', '2RH', 'sj', 'sl', 'NONE', 'invalid'):
            with self.subTest(value=invalid), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                parser.parse_args(['--dataset', 'tasks.jsonl', '--out', 'results.json', '--scaffold', invalid])

    def test_sl_selectors_use_explicit_names_without_legacy_aliases(self):
        from si2ca.strategies.sl import Strategy as SL

        self.assertIsInstance(runner.load_strategy('sl_argmin_nll'), runner.SLArgminNLL)
        self.assertEqual(SL().name, 'sl_argmin_nll_k4')
        self.assertEqual(Gain().name, 'sl_argmax_delta')
        self.assertTrue(issubclass(SL, runner.SLArgminNLL))
        self.assertTrue(issubclass(Gain, runner.SLArgminNLL))
        self.assertFalse(hasattr(runner, 'HArgminNLL'))
        self.assertNotIn('h_argmin_nll', runner.BUILTIN)
        with self.assertRaises(SystemExit):
            runner.load_strategy('h_argmin_nll')

    def test_benchmark_counts_and_assets(self):
        for benchmark,count in [('verified',500),('pro',731),('both',1231),('deepswe',113),('validation',192)]:
            rows=load_dataset(benchmark)
            self.assertEqual(len(rows),count)
            self.assertEqual(len({r['metadata']['instance_id'] for r in rows}),count)

    def test_pro_is_the_complete_public_split_with_privileged_inputs(self):
        from si2ca.runtime.swe import _swepro_instance_id

        rows = load_dataset('pro')
        self.assertEqual(Counter(r['metadata']['language'] for r in rows),
                         {'python': 266, 'go': 280, 'js': 165, 'ts': 20})
        ids = '\n'.join(sorted(r['metadata']['instance_id'] for r in rows)) + '\n'
        self.assertEqual(hashlib.sha256(ids.encode()).hexdigest(),
                         '34c6449a543beaab0ecfceaf0b63b71f8cfec5189f1f17ba9222f7a46fd96fa8')
        for row in rows:
            md = row['metadata']
            with self.subTest(task=md['instance_id']):
                self.assertTrue(md['hint'].strip())
                self.assertTrue(md['gold_patch'].startswith('diff --git '))
                self.assertTrue(md['swepro']['fail_to_pass'])
                for key in ('run_script_path', 'parser_script_path'):
                    self.assertTrue(Path(md['swepro'][key]).is_file())
                if md['language'] != 'python':
                    # Repository locks and per-task grading fixes depend on this ID.
                    self.assertEqual(_swepro_instance_id(md['swepro']), md['instance_id'])

    def test_combined_manifest_preserves_existing_tasks_and_uses_portable_assets(self):
        path = RESOURCE_ROOT / 'data/bench1231.jsonl'
        prefix = b''.join(path.read_bytes().splitlines(keepends=True)[:766])
        self.assertEqual(hashlib.sha256(prefix).hexdigest(),
                         '0bc63159f0cfe63b400d58775dcedbc392ae638dd1eab41cbca13c939d399694')
        combined = load_dataset('both')
        self.assertEqual(combined[:500], load_dataset('verified'))
        self.assertEqual(combined[500:], load_dataset('pro'))
        self.assertEqual(load_dataset('pro', limit=3), combined[500:503])
        self.assertFalse((RESOURCE_ROOT / 'data/bench766.jsonl').exists())
        for row in read_rows(path):
            for key in ('run_script_path', 'parser_script_path'):
                value = row['metadata'].get('swepro', {}).get(key)
                if value:
                    self.assertFalse(Path(value).is_absolute())

    def test_multilingual_manifest_and_validation_status(self):
        rows=load_dataset('validation')
        with (ROOT/'data/dev192_multilingual_manifest.csv').open() as f:
            manifest=list(csv.DictReader(f))
        self.assertEqual([r['metadata']['instance_id'] for r in rows],[r['instance_id'] for r in manifest])
        self.assertTrue(all(r['metadata']['benchmark']=='swebench_multilingual' for r in rows))
        self.assertTrue(all(r['metadata']['gold_patch'] for r in rows))
        self.assertEqual([r['metadata']['gold_validated'] for r in rows],
                         [r['gold_validated'].lower() == 'true' for r in manifest])
        self.assertEqual(sum(r['block']=='existing64' for r in manifest),64)

    def test_every_named_setting_resolves_offline(self):
        settings=yaml.safe_load((ROOT/'configs/experiments.yaml').read_text())['settings']
        with tempfile.TemporaryDirectory() as tok:
            for setting in settings:
                argv=['--setting',setting,'--model','policy','--base-url','http://localhost:8151/v1',
                      '--out','runs/test_'+setting,'--tokenizer-path',tok]
                if setting=='sj_strong':argv+=['--judge-model','judge']
                cfg=resolve(parser().parse_args(argv))
                self.assertNotIn('execute', cfg)
                self.assertEqual(cfg['engine'], 'strategy')
                self.assertEqual(cfg['harness_protocol'], runner.HARNESS_PROTOCOL)
                self.assertFalse(cfg['judge_batch_samples'])
                self.assertEqual(cfg['base_url'],'http://localhost:8151')
                self.assertEqual(cfg['judge_model'],'judge' if setting=='sj_strong' else 'policy')

    def test_search_baselines_route_to_multilingual_grader(self):
        for setting,strategy in [('standard','baseline'),('sj','selfguide_rubrics')]:
            cfg=resolve(parser().parse_args(['--setting',setting,'--benchmark','validation',
                '--model','policy','--base-url','http://localhost:8151','--out','runs/test']))
            self.assertEqual((cfg['engine'],cfg['strategy']),('strategy',strategy))

    def test_invalid_routes_and_counts_fail_before_execution(self):
        for extra in [['--concurrency','0'],['--n-samples','0'],['--benchmark','deepswe'],['--base-url','']]:
            argv=['--setting','sj','--model','policy','--base-url','http://localhost:8151','--out','runs/test']+extra
            with self.assertRaises(ValueError):resolve(parser().parse_args(argv))

    def test_judge_history_removes_prior_reasoning(self):
        history=[{'role':'user','content':'<pr_description>Bug</pr_description>'},
                 {'role':'assistant','content':'PRIVATE_REASONING','reasoning_content':'SECRET_THINK',
                  'tool_calls':candidate('ls')['msg']['tool_calls']},
                 {'role':'tool','content':'visible evidence'}]
        rendered=render_prefix(history,SimpleNamespace(
            prefix_tool_result_head_chars=1500, prefix_tool_result_tail_chars=1500, prefix_max_chars=80000))
        self.assertNotIn('PRIVATE_REASONING',rendered)
        self.assertNotIn('SECRET_THINK',rendered)
        self.assertIn('visible evidence',rendered)


class Selectors(unittest.IsolatedAsyncioTestCase):
    async def test_sandbox_command_drains_large_stdout_and_stderr(self):
        from si2ca.runtime.sandbox import LocalDockerSandbox
        size = 600_000
        rc, out, err = await LocalDockerSandbox._run([
            sys.executable, '-c',
            f'import sys; sys.stdout.write("o" * {size}); sys.stderr.write("e" * {size})'
        ], timeout=5)
        self.assertEqual(rc, 0)
        self.assertEqual(out, 'o' * size)
        self.assertEqual(err, 'e' * size)

    async def test_sandbox_command_timeout_is_still_reported(self):
        from si2ca.runtime.sandbox import LocalDockerSandbox
        rc, out, err = await LocalDockerSandbox._run([
            sys.executable, '-c', 'import time; time.sleep(5)'
        ], timeout=0.1)
        self.assertEqual((rc, out), (124, ''))
        self.assertIn('timed out', err)

    async def test_sandbox_command_bounds_inherited_open_pipe(self):
        from si2ca.runtime.sandbox import LocalDockerSandbox
        chunks = [b'container-id\n']

        async def read(_):
            if chunks:
                return chunks.pop()
            await asyncio.Event().wait()

        proc = SimpleNamespace(returncode=0, stdin=None, stderr=None,
                               stdout=SimpleNamespace(read=read))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=proc)):
            rc, out, err = await asyncio.wait_for(LocalDockerSandbox._run(['docker', 'run']), 5)
        self.assertEqual((rc, out, err), (0, 'container-id\n', ''))

    async def test_rubric_averaging_ignores_reported_total(self):
        scores=iter([9,1,5,7,7,7]);calls=[]
        async def judge(user,system):
            calls.append(user); value=next(scores)
            return json.dumps({**{k:value for k in runner.SelfGuideRubrics.KEYS7+('G1',)},'total':1000-value}),{}
        index,label,_=await runner.SelfGuideRubrics().select(context(),[candidate('a'),candidate('b')],judge)
        self.assertEqual(index,1)
        self.assertEqual(len(calls),6)
        self.assertEqual(label,'rubrics_score')

    async def test_no_pi_judge_has_no_gold_item(self):
        async def judge(user,system):
            self.assertNotIn('"G1"',system)
            self.assertNotIn(runner.GOLD_SLOT,user)
            return json.dumps({k:5 for k in runner.SelfGuideRubrics.KEYS7}),{}
        await runner.SelfGuideRubrics().select(context(gold=False),[candidate('a'),candidate('b')],judge)

    async def test_identical_commands_skip_judge(self):
        judge=AsyncMock()
        with patch.object(runner,'NULL_CONTROL_RATE',0):
            index,label,_=await runner.SelfGuideRubrics().select(context(),[candidate('a'),candidate('a')],judge)
        self.assertEqual(index,0); self.assertEqual(label,'identical_command'); judge.assert_not_called()

    async def test_likelihood_and_gain_make_different_choices(self):
        cands=[candidate('a',2,.5),candidate('b',5,1)]
        self.assertEqual((await runner.SLArgminNLL().select(context(),cands,None))[0],0)
        self.assertEqual((await Gain().select(context(),cands,None))[0],1)
        cands=[candidate('a'),candidate('b')]
        self.assertEqual((await runner.SLArgminNLL().select(context(),cands,None))[1],'priv_nll_unavailable_first')

    def test_discovered_window(self):
        cases=[([],1),(['ls'],2),(['ls','sed -i x a'],2),
               (['ls','sed -i x a','apply_patch x'],2),
               (['ls','sed -i x a','apply_patch x','tee file'],1)]
        for commands,wanted in cases:self.assertEqual(Discovered().plan(context(commands)),wanted)

    async def test_privileged_rescoring_never_mutates_policy_context(self):
        prefix=[{'role':'user','content':'Task'},{'role':'assistant','content':'Earlier reasoning'}]
        original=copy.deepcopy(prefix);block={'role':'user','content':'PRIVILEGED_ONLY'}
        seen=[]
        def spans(tok,messages,cand):
            seen.append(messages)
            return {'ids':[1,2,3,4],'start':2}
        def handle(request):
            payload=json.loads(request.content)
            self.assertEqual(payload['sampling_params']['max_new_tokens'],0)
            return httpx.Response(200,json={'meta_info':{'input_token_logprobs':[[-.2,3],[-.4,4]]}})
        async with httpx.AsyncClient(base_url='http://mock',transport=httpx.MockTransport(handle)) as client:
            for position in ('head','tail'):
                cands=[candidate('ls')]
                with patch('si2ca.runtime.spans.build_spans',side_effect=spans):
                    await runner.rescore_privileged(client,None,prefix,block,cands,position)
                self.assertAlmostEqual(cands[0]['mean_nll_privileged'],.3)
                self.assertEqual(seen[-1][1 if position=='head' else -1],block)
                self.assertEqual(prefix,original)

    async def test_scaffold_modes_with_mocked_model_and_sandbox(self):
        from si2ca.runtime import sandbox,swe
        original_client=httpx.AsyncClient
        generated=[];judged=[]
        def respond(request):
            payload=json.loads(request.content)
            if payload.get('tools'):
                self.assertNotIn('PRIVILEGED_ONLY',json.dumps(payload))
                generated.append(payload)
                cmd='echo '+runner.SUBMIT_MARKER+' # '+str(len(generated))
                msg=candidate(cmd)['msg']
            else:
                self.assertIn('PRIVILEGED_ONLY',json.dumps(payload));judged.append(payload)
                self.assertEqual(payload['max_tokens'],123)
                msg={'role':'assistant','content':json.dumps({k:8 for k in runner.SelfGuideRubrics.KEYS7+('G1',)})}
            return httpx.Response(200,json={'choices':[{'message':msg,'finish_reason':'tool_calls'}],
                                           'usage':{'prompt_tokens':10,'completion_tokens':5}})
        def client(*args,**kwargs):
            return original_client(*args,**kwargs,transport=httpx.MockTransport(respond))
        class FakeSandbox:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def exec(self,command,**kwargs):
                return (0, 'diff --git a/a.py b/a.py\n' if 'git diff --cached' in command else (runner.SUBMIT_MARKER if 'echo '+runner.SUBMIT_MARKER in command else ''), '')
        modes = [('SJ', runner.SelfGuideRubrics, 2, 6), ('SL', runner.SLArgminNLL, 2, 0),
                 ('SJ+SL', runner.SelfGuideRubrics, 2, 6), ('none', runner.Strategy, 1, 0)]
        for scaffold, strategy, draws, scores in modes:
            generated.clear(); judged.clear()
            with self.subTest(scaffold=scaffold), tempfile.TemporaryDirectory() as tmp:
                args=SimpleNamespace(scaffold=scaffold,inject_source='gold',inject_position='head',inject_template=None,
                    base_url='http://mock/v1',model='policy',reasoning_effort=None,temperature=1.,top_p=.95,top_k=10,
                    logprobs=False,max_gen_tokens=4096,judge_max_tokens=123,draw_retries=1,seed=42,cot_in_content=0,
                    eval_timeout=30,traj_dir=tmp,out=tmp+'/results.json',n_samples=1,tokenizer_path='not-loaded')
                row={'metadata':{'instance_id':'test','image':'never-created','problem_statement':'Fix bug',
                                  'gold_patch':'PRIVILEGED_ONLY','eval_cmd':'true'}}
                config={'agent':{'system_template':'Helpful assistant','instance_template':'<pr_description>{{task}}</pr_description>'}}
                with patch.object(sandbox,'create_sandbox',return_value=FakeSandbox()), \
                     patch.object(sandbox,'ensure_agent_user',new_callable=AsyncMock), \
                     patch.object(swe,'prepare_workspace',new_callable=AsyncMock), \
                     patch.object(swe,'evaluate',new=AsyncMock(return_value=(1.,True))), \
                     patch.object(runner.httpx,'AsyncClient',side_effect=client), \
                     patch.object(runner,'_rescore_tokenizer'), \
                     patch.object(runner,'rescore_privileged',new_callable=AsyncMock) as rescore:
                    result=await runner.run_one(row,config,args,strategy(),asyncio.Semaphore(1))
                self.assertEqual(result['reward'],1);self.assertIsNone(result['abort'])
                self.assertEqual((len(generated),len(judged)),(draws,scores))
                self.assertEqual(result['scaffold'],scaffold)
                self.assertEqual(result.get('gold_in_judge',False),scaffold in ('SJ','SJ+SL'))
                if scaffold in ('SL','SJ+SL'):
                    rescore.assert_awaited_once()
                    prefix,block=rescore.await_args.args[2:4]
                    self.assertNotIn('PRIVILEGED_ONLY',json.dumps(prefix))
                    self.assertIn('PRIVILEGED_ONLY',block['content'])
                else:
                    rescore.assert_not_awaited()
                exported=json.loads((Path(tmp)/'test.json').read_text())
                self.assertNotIn('PRIVILEGED_ONLY',json.dumps(exported['messages']))
                self.assertEqual(result['n_candidates_drawn'],draws)

    async def test_resume_preserves_explicit_scaffolds_without_repeating_completed_tasks(self):
        for scaffold in ('none', 'SJ', 'SL', 'SJ+SL'):
            with self.subTest(scaffold=scaffold), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                dataset=root/'tasks.jsonl';output=root/'results.json'
                dataset.write_text(''.join(json.dumps({'metadata':{'instance_id':iid,'gold_patch':'gold'}})+'\n'
                                           for iid in ('done','pending')))
                def record(iid, scaffold):
                    return {'instance_id':iid,'sample':0,'scaffold':scaffold,'reward':1.0,
                            'harness_protocol':runner.HARNESS_PROTOCOL,
                            'gen_prompt_tokens':1,'gen_completion_tokens':1,
                            'judge_prompt_tokens':0,'judge_completion_tokens':0,'decisions':{}}
                output.write_text(json.dumps({'strategy':'baseline','model':'policy','scaffold':scaffold,
                                              'harness_protocol':runner.HARNESS_PROTOCOL,
                                              'results':[record('done',scaffold)]}))
                args=runner.build_parser().parse_args(['--dataset',str(dataset),'--out',str(output),
                    '--model','policy','--scaffold',scaffold,'--inject-source','gold'])
                with patch.object(runner,'load_official_config',return_value={'agent':{}}), \
                     patch.object(runner,'run_one_bounded',new_callable=AsyncMock,
                                  return_value=record('pending',scaffold)) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    await runner.amain(args)
                run.assert_awaited_once()
                self.assertEqual(run.await_args.args[0]['metadata']['instance_id'],'pending')
                self.assertEqual(args.logprobs,scaffold in ('SL','SJ+SL'))
                saved=json.loads(output.read_text())
                self.assertEqual(saved['scaffold'],scaffold)
                self.assertEqual([r['scaffold'] for r in saved['results']],[scaffold,scaffold])

    async def test_invalid_scaffolds_fail_before_loading_or_sandbox_creation(self):
        from si2ca.runtime import sandbox

        for scaffold in ('R', 'H', 'RH', '2R', '2H', '2RH', 'invalid'):
            args = SimpleNamespace(scaffold=scaffold)
            with self.subTest(scaffold=scaffold), \
                 patch.object(runner, 'load_official_config') as load, \
                 patch.object(sandbox, 'create_sandbox') as create:
                with self.assertRaisesRegex(ValueError, 'Scaffold must be'):
                    await runner.amain(args)
                with self.assertRaisesRegex(ValueError, 'Scaffold must be'):
                    await runner.run_one({}, {}, args, runner.Strategy(), asyncio.Semaphore(1))
                load.assert_not_called()
                create.assert_not_called()

    async def test_resume_rejects_nonmatching_scaffolds_without_rewriting_results(self):
        for stored in ('R', 'H', 'RH', '2H', 'SL', None):
            for location in ('run', 'task'):
                with self.subTest(stored=stored, location=location), tempfile.TemporaryDirectory() as tmp:
                    dataset, output = Path(tmp)/'tasks.jsonl', Path(tmp)/'results.json'
                    dataset.write_text(json.dumps({'metadata': {'instance_id': 'done', 'gold_patch': 'gold'}})+'\n')
                    record = {'instance_id': 'done', 'sample': 0, 'scaffold': 'SJ',
                              'harness_protocol': runner.HARNESS_PROTOCOL, 'reward': 1.0}
                    saved = {'strategy': 'baseline', 'model': 'policy', 'scaffold': 'SJ',
                             'harness_protocol': runner.HARNESS_PROTOCOL, 'results': [record]}
                    (saved if location == 'run' else record)['scaffold'] = stored
                    output.write_text(json.dumps(saved))
                    before = output.read_bytes()
                    args = runner.build_parser().parse_args(['--dataset', str(dataset), '--out', str(output),
                        '--model', 'policy', '--scaffold', 'SJ', '--inject-source', 'gold'])
                    with patch.object(runner, 'load_official_config', return_value={'agent': {}}), \
                         patch.object(runner, 'run_one_bounded', new_callable=AsyncMock) as run, \
                         contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, 'choose a new --out'):
                            await runner.amain(args)
                        run.assert_not_awaited()
                    self.assertEqual(output.read_bytes(), before)


class UnifiedHarness(unittest.IsolatedAsyncioTestCase):
    async def trial(self, batches, *, strategy=None, benchmark='verified', outputs=None, fail=None):
        from si2ca.runtime import sandbox, swe, deepswe_grading
        requests, commands = [], []
        active = False
        draws = 0
        original_client = httpx.AsyncClient

        def respond(request):
            nonlocal draws
            payload = json.loads(request.content)
            requests.append((request, payload))
            generation = bool(payload.get('tools'))
            if fail == ('generation' if generation else 'judge'):
                return httpx.Response(400, json={'error': 'mock failure'})
            if generation:
                batch = batches[min(draws, len(batches)-1)]
                draws += 1
                msg = candidate(batch[0])['msg']
                msg['content'] = 'Retain this reasoning.'
                msg['tool_calls'] = [candidate(cmd)['msg']['tool_calls'][0] for cmd in batch]
            else:
                msg = {'role': 'assistant', 'content': json.dumps({k: 8 for k in runner.SelfGuideRubrics.KEYS7+('G1',)})}
            return httpx.Response(200, json={'choices': [{'message': msg, 'finish_reason': 'tool_calls'}],
                                            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})

        class Sandbox:
            async def __aenter__(self):
                nonlocal active
                active = True
                return self
            async def __aexit__(self, *unused):
                nonlocal active
                active = False
            async def exec(self, command, **kwargs):
                if kwargs.get('user') == 'agent':
                    commands.append(command)
                    command = command.removeprefix('cd /app && ')
                    return (outputs or {}).get(command, (0, runner.SUBMIT_MARKER if command.startswith('echo '+runner.SUBMIT_MARKER) else 'observed', ''))
                return (0, 'diff --git a/f b/f\n' if 'git diff --cached' in command else '', '')

        async def fresh_grade(**kwargs):
            self.assertFalse(active, 'Fresh grading must run after the solving sandbox closes')
            return 1.0, True

        async def inplace_grade(sb, md):
            self.assertTrue(active, 'DeepSWE must grade in the solving sandbox')
            return 1.0, 'OK', {'f2p_passed': 1}

        with tempfile.TemporaryDirectory() as tmp:
            args = runner.build_parser().parse_args(['--dataset', 'unused', '--out', tmp+'/results.json',
                '--traj-dir', tmp, '--model', 'policy', '--base-url', 'http://policy/v1',
                '--scaffold', 'SJ', '--inject-source', 'gold', '--judge-url', 'http://judge/v1/chat/completions',
                '--judge-model', 'judge', '--judge-temperature', '0.8', '--judge-top-p', '0.9',
                '--judge-reasoning-effort', 'high', '--request-retries', '0', '--max-turns', '3'])
            args.benchmark = benchmark
            row = {'metadata': {'instance_id': 'mock', 'image': 'never-created', 'workdir': '/app',
                               'problem_statement': 'Fix bug', 'gold_patch': 'PRIVATE_GOLD', 'eval_cmd': 'true'}}
            cfg = {'agent': {'system_template': 'Work in /testbed', 'instance_template': '<pr_description>{{task}}</pr_description> /testbed'}}
            with patch.object(sandbox, 'create_sandbox', return_value=Sandbox()), \
                 patch.object(sandbox, 'ensure_agent_user', new_callable=AsyncMock), \
                 patch.object(swe, 'prepare_workspace', new_callable=AsyncMock), \
                 patch.object(swe, 'evaluate', new=AsyncMock(side_effect=fresh_grade)) as fresh, \
                 patch.object(deepswe_grading, 'grade', new=AsyncMock(side_effect=inplace_grade)) as inplace, \
                 patch.object(runner.httpx, 'AsyncClient', side_effect=lambda *a, **kw: original_client(*a, **kw, transport=httpx.MockTransport(respond))), \
                 patch.dict(os.environ, {'SI2CA_BACKEND': 'self-hosted', 'SI2CA_JUDGE_BACKEND': 'self-hosted',
                                         'SI2CA_API_KEY': 'policy-key', 'SI2CA_JUDGE_API_KEY': 'judge-key'}):
                record = await runner.run_one(row, cfg, args, strategy or runner.Strategy(), asyncio.Semaphore(1))
            if record.get('patch_path'):
                self.assertTrue(Path(record['patch_path']).is_file())
                self.assertEqual(len(Path(record['patch_path']).read_text()), record['diff_len'])
            exported = json.loads((Path(tmp)/'mock.json').read_text())
            return record, exported, requests, commands, fresh.await_count, inplace.await_count

    async def test_multiple_tools_workdir_history_and_judge_transport(self):
        submit = 'echo '+runner.SUBMIT_MARKER
        record, exported, requests, commands, fresh, inplace = await self.trial(
            [['pwd', submit+' # a'], ['pwd', submit+' # b']], strategy=runner.SelfGuideRubrics())
        self.assertEqual((record['turns'], fresh, inplace), (1, 1, 0))
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(c.startswith('cd /app && ') for c in commands))
        self.assertEqual(record['harness_protocol'], runner.HARNESS_PROTOCOL)
        self.assertEqual(record['decisions'], {'rubrics_tie_random': 1})
        judged = [(r, p) for r, p in requests if not p.get('tools')]
        self.assertEqual(len(judged), 6, 'Candidates differing only in the second call must be judged')
        for request, payload in judged:
            self.assertEqual((request.url.host, payload['model']), ('judge', 'judge'))
            self.assertEqual(request.headers['Authorization'], 'Bearer judge-key')
            self.assertEqual((payload['temperature'], payload['top_p'], payload['reasoning_effort']), (.8, .9, 'high'))
            self.assertNotIn('n', payload)
            self.assertIn('0.105*R1', payload['messages'][0]['content'])
            self.assertIn(submit, payload['messages'][1]['content'])
        for _, payload in requests:
            if payload.get('tools'):
                self.assertNotIn('PRIVATE_GOLD', json.dumps(payload))
                self.assertNotIn('/testbed', json.dumps(payload))
        self.assertNotIn('PRIVATE_GOLD', json.dumps(exported['messages']))
        self.assertIn('Retain this reasoning.', json.dumps(exported['messages']))
        self.assertEqual(sum(m['role']=='tool' for m in exported['messages']), 2)

    async def test_submission_requires_successful_first_output_line(self):
        submit = 'echo '+runner.SUBMIT_MARKER
        invalid = 'echo prefix; '+submit
        failed = submit+'; false'
        record, _, _, commands, _, _ = await self.trial([[invalid], [failed], [submit]], outputs={
            invalid: (0, 'prefix\n'+runner.SUBMIT_MARKER, ''), failed: (1, runner.SUBMIT_MARKER, '')})
        self.assertEqual((record['turns'], len(commands), record['submitted']), (3, 3, True))

    async def test_submission_does_not_execute_later_tools_or_leave_dangling_ids(self):
        record, exported, _, commands, _, _ = await self.trial([['echo '+runner.SUBMIT_MARKER, 'must-not-run']])
        self.assertEqual(len(commands), 1)
        self.assertEqual(record['skipped_after_submit'], 1)
        assistant = next(m for m in exported['messages'] if m['role']=='assistant')
        self.assertEqual(len(assistant['tool_calls']), 1)

    async def test_deepswe_standard_and_sj_share_loop_and_inplace_grading(self):
        for strategy in (runner.Strategy(), runner.SelfGuideRubrics(), Discovered()):
            with self.subTest(strategy=strategy.name):
                record, _, _, _, fresh, inplace = await self.trial(
                    [['echo '+runner.SUBMIT_MARKER]], strategy=strategy, benchmark='deepswe')
                self.assertEqual((fresh, inplace, record['grade_status']), (0, 1, 'OK'))
                self.assertEqual(record['grading_protocol'], 'deepswe_in_place')

    async def test_transport_failures_are_incomplete_and_exported(self):
        for stage in ('generation', 'judge'):
            record, exported, _, commands, _, _ = await self.trial(
                [['echo a'], ['echo b']], strategy=runner.SelfGuideRubrics(), fail=stage)
            self.assertEqual(record['execution_status'], 'infrastructure_failure')
            self.assertFalse(summary({('mock', 0): record}, 1)['final'])
            self.assertEqual(exported['result']['harness_protocol'], runner.HARNESS_PROTOCOL)
            self.assertEqual(commands, [])

    async def test_identical_and_tied_choices_use_seeded_rng_without_null_control(self):
        judge = AsyncMock(return_value=(json.dumps({k: 5 for k in runner.SelfGuideRubrics.KEYS7+('G1',)}), {}))
        ctx = context()
        ctx.rng = SimpleNamespace(choice=lambda xs: xs[-1])
        index, label, _ = await runner.SelfGuideRubrics().select(ctx, [candidate('a'), candidate('a')], judge)
        self.assertEqual((index, label), (1, 'identical_command'))
        judge.assert_not_awaited()
        index, label, _ = await runner.SelfGuideRubrics().select(ctx, [candidate('a'), candidate('b')], judge)
        self.assertEqual((index, label), (1, 'rubrics_tie_random'))
        for selector in (runner.SLArgminNLL(), Gain()):
            index, label, _ = await selector.select(ctx, [candidate('a', 2, 1), candidate('a', 2, 1)], None)
            self.assertEqual((index, label), (1, 'identical_command'))
            index, label, _ = await selector.select(ctx, [candidate('a', 2, 1), candidate('b', 2, 1)], None)
            self.assertEqual(index, 1)
            self.assertTrue(label.endswith('tie_random'))

    def test_malformed_tool_batches_are_rejected_as_a_whole(self):
        valid = candidate('pwd')['msg']['tool_calls'][0]
        for calls in ({'invalid': 'batch'}, [valid, {'function': 'invalid'}],
                      [valid, {'function': {'name': 'not-bash', 'arguments': '{}'}}],
                      [valid, {'function': {'name': 'bash', 'arguments': '{'}}]):
            self.assertEqual(runner.parse_actions({'tool_calls': calls}), [])

    def test_discovered_uses_the_shared_window_and_sj_selector(self):
        from si2ca.runtime.protocol import early_commit_window
        self.assertIs(Discovered.select.__globals__['SelfGuideRubrics'], runner.SelfGuideRubrics)
        for edits in range(6):
            commands = ['ls']+['sed -i x f']*edits
            self.assertEqual(Discovered().gate(context(commands)), early_commit_window(commands))

    async def test_unversioned_resume_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'results.json'
            output.write_text('{"results": []}')
            dataset = Path(tmp)/'tasks.jsonl'
            dataset.write_text('{"metadata": {"instance_id": "mock"}}\n')
            args = runner.build_parser().parse_args(['--dataset', str(dataset), '--out', str(output)])
            with patch.object(runner, 'load_official_config', return_value={'agent': {}}), \
                 patch.object(runner, 'run_one_bounded', new_callable=AsyncMock) as run:
                with self.assertRaisesRegex(ValueError, 'unversioned harness'):
                    await runner.amain(args)
                run.assert_not_awaited()
            self.assertEqual(output.read_text(), '{"results": []}')

    async def test_resume_retries_infrastructure_failures_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, dataset = Path(tmp)/'results.json', Path(tmp)/'tasks.jsonl'
            dataset.write_text(''.join(json.dumps({'metadata': {'instance_id': iid}})+'\n'
                                       for iid in ('complete', 'failed')))
            def record(iid):
                return {'instance_id': iid, 'harness_protocol': runner.HARNESS_PROTOCOL,
                        'reward': 0, 'turns': 1, 'gen_prompt_tokens': 1, 'gen_completion_tokens': 1,
                        'judge_prompt_tokens': 0, 'judge_completion_tokens': 0}
            output.write_text(json.dumps({'harness_protocol': runner.HARNESS_PROTOCOL,
                'model': 'policy', 'strategy': 'baseline', 'scaffold': 'none',
                'results': [record('complete'), record('failed') | {'abort': 'all_draws_failed'}]}))
            args = runner.build_parser().parse_args(['--dataset', str(dataset), '--out', str(output), '--model', 'policy'])
            with patch.object(runner, 'load_official_config', return_value={'agent': {}}), \
                 patch.object(runner, 'run_one_bounded', new_callable=AsyncMock, return_value=record('failed')) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                await runner.amain(args)
            run.assert_awaited_once()
            self.assertEqual(run.await_args.args[0]['metadata']['instance_id'], 'failed')
            saved = json.loads(output.read_text())['results']
            self.assertEqual(len(saved), 2)
            self.assertTrue(summary({(r['instance_id'], 0): r for r in saved}, 2)['final'])

    def test_api_exit_codes_do_not_make_a_final_result_or_shrink_denominator(self):
        records = {('a',0): {'reward': 1, 'turns': 2}, ('b',0): {'reward': 1, 'turns': 3, 'exit_code': 1}}
        result = summary(records, 2)
        self.assertEqual((result['solved'], result['expected_trials'], result['infrastructure_failures']), (1, 2, 1))
        self.assertEqual((result['turns_mean'], result['turns_trials']), (2, 1))
        self.assertFalse(result['final'])
        old = {(str(i),0): {'reward': 0, 'turns': 10} for i in range(192)}
        new = {key: value | {'harness_protocol': runner.HARNESS_PROTOCOL} for key, value in old.items()}
        with self.assertRaisesRegex(ValueError, 'matching harness versions'):
            decide(old, new, context_audit=True)


class AccountingAndTraining(unittest.TestCase):
    def test_sft_launches_directly_with_stubbed_training_commands(self):
        # Shell-only fixture: neither Ray nor the GPU training interpreter runs.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            command_log = root / 'commands.log'
            for command in ('ray', 'python'):
                stub = bin_dir / command
                stub.write_text('#!/bin/bash\nprintf "%s\\n" "${0##*/} $*" >> "$SI2CA_TEST_COMMAND_LOG"\n')
                stub.chmod(0o755)
            for file in ('sft.jsonl', 'hf/config.json', 'ref/latest_checkpointed_iteration.txt'):
                path = root / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n')
            (root / 'megatron/megatron/training').mkdir(parents=True)
            env = os.environ | {
                'PATH': f'{bin_dir}:/usr/bin:/bin', 'SI2CA_TEST_COMMAND_LOG': str(command_log),
                'SFT_DATA': str(root / 'sft.jsonl'), 'STUDENT_HF': str(root / 'hf'),
                'STUDENT_REF': str(root / 'ref'), 'SAVE_DIR': str(root / 'save'),
                'MEGATRON_PATH': str(root / 'megatron'), 'RESUME': '0',
            }
            subprocess.run(['/bin/bash', str(ROOT / 'training/sft.sh')], env=env,
                           capture_output=True, text=True, check=True, timeout=10)
            calls = command_log.read_text().splitlines()
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[0].startswith('ray start --head '))
            self.assertEqual(calls[1], 'ray status --address=127.0.0.1:6380')
            self.assertTrue(calls[2].startswith(f'python -u {ROOT}/training/backend/train_async.py '))
            self.assertIn(f'--prompt-data {root}/sft.jsonl', calls[2])

    def test_fixed_denominator_and_pending_validation(self):
        records={('x',0):{'reward':1.,'turns':10,'gold_validated':False}}
        result=summary(records,2)
        self.assertEqual(result['resolve_pct'],50)
        self.assertEqual(result['missing_trials'],1)
        self.assertFalse(result['final'])
        self.assertEqual(result['unvalidated_grading_chains'],1)

    def test_retry_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'results.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in [{'instance_id':'a','reward':0}, {'instance_id':'a','reward':1}]))
            self.assertEqual(summary(load_results(p),1)['solved'],1)

    def test_search_rejects_missing_context_audit_or_mismatched_tasks(self):
        old={(str(i),0):{'reward':int(i<99),'turns':100} for i in range(192)}
        new={(str(i),0):{'reward':int(i<106),'turns':100} for i in range(192)}
        self.assertFalse(decide(old,new)['accept'])
        self.assertTrue(decide(old,new,context_audit=True)['accept'])
        new.pop(('191',0))
        with self.assertRaises(ValueError):decide(old,new,context_audit=True)

    def test_training_dynamic_entrypoint_is_present(self):
        self.assertTrue((ROOT/'training/backend/slime/rollout/data_source.py').is_file())
        self.assertTrue((ROOT/'training/backend/slime/rollout/sft_rollout.py').is_file())

    def test_cleaner_rejects_malformed_tool_calls(self):
        from training.prepare import normalize
        self.assertIsInstance(normalize([candidate('ls')['msg']])[0]['tool_calls'][0]['function']['arguments'],dict)
        bad=candidate('ls')['msg'];bad['tool_calls'][0]['function']['name']='invented'
        with self.assertRaises(ValueError):normalize([bad])

    def test_export_preserves_valid_eos_order_independently(self):
        from training.finalize_export import finalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe'}))
            (root/'tokenizer.json').write_text(json.dumps({'added_tokens':[
                {'content':'<|im_end|>','id':248044},{'content':'<|endoftext|>','id':248046}]}))
            (root/'generation_config.json').write_text(json.dumps({'eos_token_id':[248046,248044]}))
            finalize(root)
            self.assertEqual(set(json.loads((root/'generation_config.json').read_text())['eos_token_id']),{248044,248046})



class PromptAssets(unittest.TestCase):
    def test_agent_templates_are_owned_by_the_prompt_directory(self):
        from si2ca.prompts import PROMPT_DIR, AGENT_TEMPLATES, MODEL_TEMPLATES
        for name, text in {**AGENT_TEMPLATES, **MODEL_TEMPLATES}.items():
            self.assertEqual(text, (PROMPT_DIR / ('agent_' + name.removesuffix('_template') + '.md')).read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'swebench.yaml'
            path.write_text(yaml.safe_dump({'agent': {'system_template': 'external override', 'step_limit': 99},
                                           'model': {'format_error_template': 'external override'},
                                           'environment': {'timeout': 17}}))
            with patch('subprocess.run', return_value=SimpleNamespace(stdout=str(path), stderr='')):
                config = runner.load_official_config(sys.executable)
        self.assertEqual(config['agent']['system_template'], AGENT_TEMPLATES['system_template'])
        self.assertEqual(config['model'], MODEL_TEMPLATES)
        self.assertEqual(config['agent']['step_limit'], 99)
        self.assertEqual(config['environment']['timeout'], 17)

    def test_bash_tool_schema_has_one_asset_for_all_consumers(self):
        from si2ca.prompts import PROMPT_DIR
        from si2ca.runtime import spans
        expected = json.loads((PROMPT_DIR / 'bash_tool.json').read_text())
        for module in (spans, runner):
            self.assertEqual(module.BASH_TOOL, expected)
        self.assertIsNot(spans.BASH_TOOL, runner.BASH_TOOL)

    def test_prompt_assets_are_centralized_and_packaged(self):
        from si2ca.prompts import PROMPT_DIR, MESSAGES
        from si2ca.runtime import privileged_prompt
        self.assertTrue((PROMPT_DIR / "rubrics.json").is_file())
        self.assertTrue(all(p.parent == PROMPT_DIR for p in privileged_prompt._TEMPLATE.values()))
        self.assertFalse((PROMPT_DIR.parent / 'runtime/rubrics.json').exists())
        self.assertFalse(list((PROMPT_DIR.parent / 'runtime/prompts').glob('*.md')))
        self.assertEqual(set(MESSAGES), {'family', 'harness', 'judge', 'hints', 'shim'})
        self.assertTrue(all(p.read_text(encoding='utf-8') for p in PROMPT_DIR.glob('*.md')))

    def test_literal_reference_patch_is_not_reinterpreted(self):
        from si2ca.prompts import MESSAGES
        from si2ca.runtime import privileged_prompt
        patch_text = 'diff --git a/f b/f\n+{{task}} {% if x %} {formula}\n'
        for position in ('head', 'tail', 'hint'):
            message, metadata = privileged_prompt.gold_message(patch_text, 10000, position)
            self.assertIn(patch_text, message['content'])
        self.assertIn(patch_text, MESSAGES['judge']['reference_solution'].format(patch=patch_text))

    def test_rendered_prompts_match_pre_migration_snapshot(self):
        import hashlib
        from jinja2 import Template
        from si2ca.prompts import AGENT_TEMPLATES, MODEL_TEMPLATES
        from si2ca.runtime import strategy, privileged_prompt
        from si2ca import make_hints
        # Hashes recorded before extraction, not regenerated from the new assets.
        out = {}
        templates = AGENT_TEMPLATES | MODEL_TEMPLATES
        for name, values in (
            ('system_template', {}),
            ('instance_template', {'task': 'Fix café: {{x}} {% y %} {formula}'}),
            ('observation_template', {'output': {'exception_info':'oops', 'returncode':1,'output':'x'*12001}}),
            ('format_error_template', {'error':'bad arguments', 'finish_reason':'length'}),
            ('format_error_template', {'error':'bad arguments', 'finish_reason':'stop'}),
        ):
            out[f'agent.{name.upper()}.{values.get("finish_reason", "")}'] = Template(templates[name]).render(**values)
        out['agent.tool'] = runner.BASH_TOOL
        for placement in ('head','tail','hint'):
            for size in (10, 9000):
                out[f'privileged.{placement}.{size}'] = privileged_prompt.gold_message('diff --git a/f b/f\n+{{x}} {% y %}\n'+'a'*size, 300, placement)
        out['hint'] = make_hints.TEMPLATE.replace('{{problem_statement}}','Issue {{x}}').replace('{{gold_patch}}','Patch {%x%}')
        out['hint_retry'] = make_hints.RETRY_NOTE % (10,'below',700,1100,900)
        out['strategy_simple'] = strategy.SelfGuideK2._score_prompt(SimpleNamespace(prior_commands=['cat f'],last_observation='result'), {'command':'edit {x}'})
        config = strategy.load_official_config(sys.executable)
        out['strategy_config'] = {section:{k:v for k,v in config[section].items() if k.endswith('_template')} for section in ('agent','model')}
        expected = {
            "hint": "ba81b40dfd76e1f9533ce29e1ec94f8a2c41c1ba412904d7f4c66af21ab30cf4",
            "hint_retry": "24cc9801f78cfe88e864b45ce5ae21fa1a84f2f09b058c44d39a87da67a53fad",
            "privileged.head.10": "b6b6261851eee50679bcaf9e1d03e4778b4c0e4da9b5e36377bee66c10fab032",
            "privileged.head.9000": "e132d6963e7fb1af4955fb461ccb1bf6ab8ff6e88e51c4bdff5f3526f43d7549",
            "privileged.hint.10": "f736325605fcbca57088e03015d205e9051b1f6f8b75d2357d3a78ae9c3401ec",
            "privileged.hint.9000": "8f17a003518851d58f64be39e9f553feee58ef048a222e212fe056166e74593c",
            "privileged.tail.10": "fb6fd12a5e23370aae2db5b352a974749ce97d2f883d0beacc0cb4a9cbae714c",
            "privileged.tail.9000": "ece38aa1709c48411963491f607eba84550fee59fe8e47abf7d8d4b44689e0a5",
            "agent.FORMAT_ERROR_TEMPLATE.length": "fb6deb23f6f3c5d147d3763c39f1ab9a777bb42278f525d6812b692aecfce278",
            "agent.FORMAT_ERROR_TEMPLATE.stop": "3b64ebd71b79e7b13c80c0b8c173f1cbd0447f007fa9d7c9a9e9fe2673509f93",
            "agent.INSTANCE_TEMPLATE.": "58a92f0050baeda75ccd53e3870bae5a31ff0922d138d752b4ed2270f4d478ad",
            "agent.OBSERVATION_TEMPLATE.": "c10a5f0b9467ff75b14a7bdf73a843f37e8de297661eba9ed76e3ccb65d45794",
            "agent.SYSTEM_TEMPLATE.": "2f8085311d8032c9999a368436c61d10793f401750438fc9ee80aa8fa4ce68ff",
            "agent.tool": "e7d0cb824728a0e677d5d6521845f5b87f5907091a0538310ae964dfdaee437c",
            "strategy_config": "edb66b7023b0e699b9c41f14f99c59a121091b20b6ee5c87af3178e2c800c0a3",
            "strategy_simple": "7e0ad80f1987ef85225025ea8b7c133b63ed4d16f2525f2c36a5d070d8369d53"
        }
        self.assertEqual(set(out), set(expected))
        for name, rendered in out.items():
            with self.subTest(prompt=name):
                digest = hashlib.sha256(json.dumps(rendered, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, expected[name])


if __name__=='__main__':unittest.main()
