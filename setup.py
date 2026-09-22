"""Include the single source copy of experiment inputs in binary distributions."""
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent
        files = [source / "configs/experiments.yaml", source / "data/assets.tar.gz"]
        files += sorted((source / "data").glob("*.jsonl"))
        files += sorted((source / "data").glob("*.csv"))
        files += sorted((source / "skills").glob("*.md"))
        for path in files:
            target = Path(self.build_lib) / "si2ca/resources" / path.relative_to(source)
            self.mkpath(str(target.parent))
            self.copy_file(str(path), str(target))


setup(cmdclass={"build_py": BuildPy})
