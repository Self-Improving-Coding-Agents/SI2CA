# Publishing `si2ca`

The workflows are configured, but the package is **not yet published on PyPI**. A Git push alone does not make `pip install si2ca` available.

## Automated checks

[Python checks](../.github/workflows/ci.yml) runs on pushes to `main`, pull requests and manual runs from the Actions tab. It installs the clients in Python 3.12, checks dependencies, runs focused Pylint checks and the offline test suite, builds the source distribution and wheel, and tests the installed wheel outside the checkout. The checked packages are downloadable as the `python-distributions` artifact for seven days.

The Pylint checks cover syntax errors, undefined variables and variables used before assignment, not a full style-score policy. These jobs use GitHub-hosted CPU runners; they do not serve models, run benchmarks or call paid model APIs.

## One-time PyPI setup

Sign in to your own [PyPI account](https://pypi.org/account/register/), complete its email and two-factor authentication setup, then open [Publishing](https://pypi.org/manage/account/publishing/). Add a **pending GitHub publisher** with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `si2ca` |
| Owner | `Self-Improving-Coding-Agents` |
| Repository name | `SI2CA` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

If you configured a publisher before the organization transfer, add a publisher with the owner above before the next release. A GitHub transfer does not update PyPI's trusted publisher settings.

In GitHub, create **Settings → Environments → New environment → `pypi`**. If available for your repository, add a required reviewer so a maintainer approves each upload. Ensure GitHub Actions is enabled and the actions referenced by the workflows are allowed.

This uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/): no PyPI password or API token needs to be stored in GitHub or shared in chat. A pending publisher does not reserve the package name; PyPI checks availability at the first upload. If you already own the project on PyPI, add the same publisher under that project's Publishing settings instead.

## Publish a release

**1.** Finish and push the release changes to `main`, and wait for **Python checks** to pass. The GitHub tag must match `v` followed by `project.version` in `pyproject.toml` (currently `v0.2.6`). Each later release needs a new version; PyPI files cannot be overwritten.

**2.** Open **GitHub → Releases → Draft a new release**, select or create that version tag on the intended `main` commit, add release notes, then choose **Publish release**. Saving a draft does not upload anything.

**3.** [Publish Python package](../.github/workflows/publish.yml) reruns the checks for that release, verifies the version tag, and uploads only the checked packages. Approve the `pypi` environment job if a reviewer is configured. After it succeeds, confirm the version on [PyPI](https://pypi.org/project/si2ca/) before announcing it.

```bash
# Install the package after its first successful PyPI release.
python -m pip install si2ca
# Check the installed command.
si2ca --help
```

The wheel includes prompts, configs, benchmark manifests, grading assets and both search skills. Model weights and generated trajectories are excluded; the training backend remains in the source checkout. Use `si2ca[search]` for the Multilingual grader or `si2ca[serve]` for compatible NVIDIA SGLang serving.

## Optional supply-chain provenance

The publish action includes [PyPI digital attestations](https://github.com/pypa/gh-action-pypi-publish#generating-and-uploading-attestations) by default with Trusted Publishing. A separate SLSA Generic generator workflow is not required for installation or publication, and is not configured here. These attestations alone are not a claim of SLSA Level 3 compliance.
