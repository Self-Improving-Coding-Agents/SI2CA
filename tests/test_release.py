"""Source-checkout regression checks for curation scope and author-supplied tables."""
import ast
from html import unescape
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NUM_GPUS = 0


def table_values(readme, anchor, model_prefix):
    section = readme.split(f'<a id="{anchor}"></a>', 1)[1].split('\n##', 1)[0]
    return [[cell.split(r'\,', 1)[0].strip(' $`').replace(r'\pm', '±') for cell in line.split('|')[3:-1]]
            for line in section.splitlines() if line.startswith(f'| {model_prefix}')]


class Release(unittest.TestCase):
    def test_repository_links_and_publisher_follow_the_organization(self):
        import tomllib

        repository = 'Self-Improving-Coding-Agents/SI2CA'
        url = f'https://github.com/{repository}'
        package = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertEqual(package['project']['urls']['Repository'], url)
        self.assertEqual(package['project']['urls']['Documentation'], f'{url}/blob/main/README.md')
        publisher = (ROOT / '.github/workflows/publish.yml').read_text()
        self.assertIn(f"if: github.repository == '{repository}'", publisher)
        self.assertIn('| Owner | `Self-Improving-Coding-Agents` |',
                      (ROOT / 'docs/PUBLISHING.md').read_text())
        self.assertIn(f'repository-code: "{url}"', (ROOT / 'CITATION.cff').read_text())
        readme = (ROOT / 'README.md').read_text()
        self.assertIn(f'git@github.com:{repository}.git', readme)
        self.assertIn('repos=Self-Improving-Coding-Agents%2FSI2CA', readme)
        provenance = json.loads((ROOT / 'docs/provenance.json').read_text())
        archive = provenance['research_archive']
        archive_repository = f'{repository}-Visualization'
        self.assertEqual(archive['repository'], archive_repository)
        self.assertEqual(archive['url'],
                         f"https://github.com/{archive_repository}/tree/{archive['commit']}/paper_data")

    def test_readme_header_uses_the_supplied_logo_and_two_line_wordmark(self):
        import xml.etree.ElementTree as ET

        header = (ROOT / 'README.md').read_text().split('</h1>', 1)[0]
        self.assertIn('<h1 align="center">', header)
        self.assertIn('src="docs/static/images/logo-transparent.png"', header)
        self.assertIn('src="docs/static/images/wordmark.svg"', header)
        self.assertIn('alt="(Self-Improving)² Coding Agents"', header)
        self.assertNotIn('&nbsp;', header)
        self.assertTrue((ROOT / 'data/assets/logo.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        logo = (ROOT / 'docs/static/images/logo-transparent.png').read_bytes()
        self.assertTrue(logo.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(logo[25], 6)  # PNG IHDR color type: RGBA.
        wordmark = ET.parse(ROOT / 'docs/static/images/wordmark.svg').getroot()
        self.assertEqual([node.text for node in wordmark.iter('{http://www.w3.org/2000/svg}text')],
                         ['(Self-Improving)²', 'Coding Agents'])
        group = wordmark.find('{http://www.w3.org/2000/svg}g')
        self.assertEqual(group.get('text-anchor'), 'middle')
        self.assertEqual([node.get('x') for node in group], ['178', '168.1'])

    def test_local_runtime_imports_resolve_without_legacy_drivers(self):
        package = ROOT / 'si2ca'
        modules = {}
        for path in package.rglob('*.py'):
            parts = path.relative_to(ROOT).with_suffix('').parts
            name = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
            modules[name] = path
        missing = []
        for name, path in modules.items():
            parent = name if path.name == '__init__.py' else name.rpartition('.')[0]
            for node in ast.walk(ast.parse(path.read_text())):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        prefix = parent.split('.')[:len(parent.split('.')) - node.level + 1]
                        module = '.'.join(prefix + ([module] if module else []))
                    imports = [module]
                for module in imports:
                    if module.split('.')[0] == 'si2ca' and module not in modules:
                        missing.append(f'{path.relative_to(ROOT)}:{node.lineno}: {module}')
        self.assertEqual(missing, [])
        for module in ('minimal', 'legacy_minimal', 'self_judge', 'oracle', 'common',
                       'deepswe_standard', 'deepswe_sj', 'family_rubrics',
                       'classify_actions_v2', 'turn_stats', 'traj_io'):
            self.assertNotIn(f'si2ca.runtime.{module}', modules)
        self.assertFalse(any(name.startswith('si2ca.runtime.branch_select') for name in modules))
        self.assertIn('si2ca.runtime.deepswe_grading', modules)

    def test_authored_repository_text_is_english(self):
        checker = runpy.run_path(str(ROOT / 'scripts/check_english.py'))
        report = checker['audit'](ROOT)
        self.assertEqual(report['violations'], [])
        self.assertTrue(any(row['file'] == 'si2ca/runtime/strategy.py'
                            and row['scope'] == 'authored text' for row in report['files']))
        # The research-material exception must not hide arbitrary new code.
        self.assertIsNone(checker['RAW_ARCHIVE'].fullmatch('paper_data/new_module.py'))
        self.assertNotIn('data/new_script.py', checker['RAW_INPUTS'])

    def test_language_check_detects_encoded_text_without_rejecting_model_tokens(self):
        checker = runpy.run_path(str(ROOT / 'scripts/check_english.py'))
        slash = chr(92)
        entity = chr(38) + chr(35)
        samples = [chr(0x4E2D), slash + 'u4e2d', slash + 'u{4e2d}',
                   slash + 'U00020000', slash + 'uD840' + slash + 'uDC00',
                   slash + 'N{CJK UNIFIED IDEOGRAPH-4E2D}', entity + '20013;', entity + 'x4e2d;',
                   chr(0xFF0C)]
        for sample in samples:
            with self.subTest(sample=ascii(sample)):
                self.assertIsNotNone(checker['CHINESE'].search(checker['decoded_text'](sample)))
        model_token = '<' + chr(0xFF5C) + 'Assistant' + chr(0xFF5C) + '>'
        self.assertIsNone(checker['CHINESE'].search(model_token))

    def test_language_audit_checks_new_files_and_limits_research_exceptions(self):
        checker = runpy.run_path(str(ROOT / 'scripts/check_english.py'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '--quiet', directory], check=True)
            (root / 'data').mkdir()
            (root / 'data/bench1231.jsonl').write_text(chr(0x4E2D), encoding='utf-8')
            (root / 'data/new_script.py').write_text('# ' + chr(0x4E2D), encoding='utf-8')
            (root / 'new.bin').write_bytes(bytes([255, 254]))
            report = checker['audit'](root)
            self.assertEqual({(v['file'], v['reason']) for v in report['violations']}, {
                ('data/new_script.py', 'Chinese text'), ('new.bin', 'Unreviewed non-UTF-8 file'),
            })
            self.assertIn({'file': 'data/bench1231.jsonl', 'scope': 'preserved research material'},
                          report['files'])

    def test_deepswe_labels_have_no_normal_wrap_points(self):
        readme = (ROOT / 'README.md').read_text()
        section = readme.split('<a id="deepswe--113-tasks"></a>', 1)[1].split('\n##', 1)[0]
        rows = [line.split('|')[1:-1] for line in section.splitlines() if line.startswith('| GPT')]
        labels = [[unescape(cell.strip()) for cell in row[:2]] for row in rows]
        self.assertEqual([[cell.replace('\u2011', '-').replace('\xa0', ' ') for cell in row] for row in labels], [
            ['GPT-5.6-Luna-Max', 'Standard'],
            ['GPT-5.6-Luna-Max', 'Self-judgement'],
            ['GPT-5.6-Terra-xHigh', 'Standard'],
            ['GPT-5.6-Terra-xHigh', 'SJ, judge=xHigh'],
        ])
        for row in labels:
            for cell in row:
                self.assertFalse(any(char in cell for char in (' ', '-', '\n')))

    def test_result_values_and_colored_changes_share_one_inline_expression(self):
        readme = (ROOT / 'README.md').read_text()
        cells = [cell.strip() for line in readme.splitlines()
                 if line.startswith(('| Qwen', '| GPT')) for cell in line.split('|')[3:-1]
                 if r'\color' in cell]
        self.assertEqual(len(cells), 40)
        for cell in cells:
            self.assertRegex(cell, r'^\$`\d+(?:\.\d+)?(?: \\pm \d+(?:\.\d+)?)?\\,\\color\{(?:green|red)\}\{\(\\(?:uparrow|downarrow) [\d.]+\)\}`\$$')
        self.assertEqual(sum(r'\color{green}' in cell for cell in cells), 39)
        self.assertIn(r'$`474\,\color{red}{(\uparrow 9)}`$', cells)

    def test_local_training_imports_remain_present(self):
        root = ROOT / 'training/backend'
        missing = []
        for path in root.rglob('*.py'):
            package = path.relative_to(root).parts[:-1]
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        module = '.'.join((*package[:len(package)-node.level+1], *module.split('.'))).rstrip('.')
                    modules = [module]
                for module in modules:
                    if module.split('.')[0] not in ('slime', 'slime_plugins'):
                        continue
                    target = root.joinpath(*module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        missing.append(f'{path}:{node.lineno}: {module}')
        self.assertEqual(missing, [])
        converters = root / 'slime/backends/megatron_utils/megatron_to_hf'
        self.assertEqual({p.name for p in converters.glob('*.py')}, {'__init__.py', 'qwen3_5.py'})
        self.assertFalse((root / 'slime/agent/sandbox.py').exists())

    def test_corrected_pro_table_and_unchanged_verified(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertEqual(table_values(readme, 'swe-bench-verified--500-tasks', 'Qwen'), [
            ['65.8', '85.0', '76', '144'], ['70.0', '73.7', '67', '127'],
            ['66.8', '76.9', '68', '130'], ['67.0', '72.1', '64', '122'],
            ['71.0', '68.8', '61', '116'], ['69.8', '69.5', '62', '118']])
        self.assertEqual(table_values(readme, 'swe-bench-pro--731-tasks', 'Qwen'), [
            ['46.0', '89.4', '80', '153'], ['53.2', '79.3', '72', '133'],
            ['51.2', '77.5', '68', '134'], ['48.0', '84.6', '80', '135'],
            ['58.5', '80.2', '74', '130'], ['53.8', '69.9', '64', '117']])
        self.assertRegex(readme, r'(?m)^### .*SWE-bench Pro — 731 tasks ')
        self.assertNotIn('266 Pro-Python tasks', readme)

    def test_archived_research_is_separate_from_runtime_inputs(self):
        for name in ('paper_data', 'visualization'):
            self.assertFalse((ROOT / name).exists(), name)
        for name in ('bench1231.jsonl', 'deepswe113.jsonl', 'dev192_multilingual.jsonl', 'assets.tar.gz'):
            self.assertTrue((ROOT / 'data' / name).is_file(), name)
        readme = (ROOT / 'README.md').read_text()
        archive_url = 'https://github.com/Self-Improving-Coding-Agents/SI2CA-Visualization'
        self.assertIn(archive_url, readme)
        self.assertIn(f'{archive_url}/blob/main/paper_data/README.md',
                      (ROOT / 'docs/HARNESS.md').read_text())
        for path in ('README.md', 'docs/HARNESS.md', 'docs/RESULTS.md'):
            text = (ROOT / path).read_text()
            self.assertIn(archive_url, text)
            self.assertNotIn('SI2CA/tree/visualization', text)
            self.assertNotIn('SI2CA/blob/visualization', text)
        provenance = json.loads((ROOT / 'docs/provenance.json').read_text())
        archive = provenance['research_archive']
        self.assertEqual(archive['branch'], 'main')
        self.assertRegex(archive['commit'], r'^[0-9a-f]{40}$')
        self.assertEqual(len(archive['files']), 59)
        self.assertEqual(len({row['file'] for row in archive['files']}), 59)
        for row in archive['files']:
            self.assertTrue(row['file'].startswith('paper_data/'))
            self.assertGreater(row['bytes'], 0)
            for field in ('source_sha256', 'archive_sha256'):
                self.assertRegex(row[field], r'^[0-9a-f]{64}$')
            if row['file'] != 'paper_data/README.md':
                self.assertEqual(row['source_sha256'], row['archive_sha256'])
        self.assertFalse(any(row['file'].startswith(('paper_data/', 'visualization/'))
                             for row in provenance['release_files']))

    def test_docker_is_with_installation_and_deepswe_includes_turns(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertLess(readme.index('<a id="docker">'), readme.index('<a id="one-command-serve-and-run">'))
        self.assertIn('We also recommend using Docker', readme)
        self.assertEqual(table_values(readme, 'deepswe--113-tasks', 'GPT'), [
            ['60.2', '246.8', '173', '465', '1'],
            ['64.6', '223.0', '150', '474', '1'],
            ['64.4 ± 2.0', '59.3', '52', '98', '4'],
            ['67.5 ± 2.5', '52.3', '46', '86', '4']])
        self.assertNotIn('--model qwen122', readme)


if __name__ == '__main__':
    unittest.main()
