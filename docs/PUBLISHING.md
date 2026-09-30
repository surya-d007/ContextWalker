# Publishing ContextWalker

ContextWalker publishes to PyPI from GitHub Releases using PyPI Trusted
Publishing. The workflow does not require a stored API token.

## One-time setup

### 1. Create the GitHub environment

In the GitHub repository, open **Settings → Environments**, create an
environment named `pypi`, and add deployment protection or a required reviewer
when appropriate.

### 2. Configure the PyPI trusted publisher

Open <https://pypi.org/manage/account/publishing/> and register a pending GitHub
publisher with these exact values:

```text
PyPI project name: contextwalker
GitHub owner: surya-d007
GitHub repository: ContextWalker
Workflow filename: publish.yml
Environment name: pypi
```

The pending publisher creates the PyPI project on the first successful release.
It does not reserve the project name before that release.

## Local preflight

Run these commands from the repository root:

```bash
python -m pip install --upgrade build twine
python -m unittest discover -s tests -v
python -m build
python -m twine check --strict dist/*
```

Optionally install the newly built wheel in a fresh virtual environment before
publishing.

## Publish a release

Confirm that the version in `pyproject.toml` and `__version__` in
`src/contextwalker/__init__.py` agree. Commit and push all release changes, then
create a matching tag. For example, to publish version 0.2.1:

```bash
git tag -a v0.2.1 -m "ContextWalker 0.2.1"
git push origin v0.2.1
```

On GitHub, create a release using the matching tag and publish it. Publishing
the GitHub release starts `.github/workflows/publish.yml`. The workflow tests
the project, builds its wheel and source distribution, validates the metadata,
and publishes through the configured trusted publisher.

## Verify the release

After the workflow succeeds:

```bash
python -m pip install contextwalker
contextwalker
```

The project page will be available at <https://pypi.org/project/contextwalker/>.

## Later releases

PyPI does not permit replacing an existing release file. For every release:

1. Choose a new version number.
2. Update both `pyproject.toml` and `src/contextwalker/__init__.py`.
3. Commit and push the changes.
4. Create a matching `vX.Y.Z` tag.
5. Publish a GitHub release from that tag.
