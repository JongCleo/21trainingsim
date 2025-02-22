## Scaffolding

```bash
uv init
uv venv
uv add fastapi pydantic uvicorn python-dotenv loguru openai
uv add --dev pytest
# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS

# write [tool.pytest.ini_options] to pyproject.toml for testing setup
echo "[tool.pytest.ini_options]" >> pyproject.toml
echo "pythonpath = \"src\"" >> pyproject.toml

# this is for testing setup
echo "asyncio_mode = \"auto\"" >> pyproject.toml
echo "asyncio_default_fixture_loop_scope = \"function\"" >> pyproject.toml

# this is for ruff setup
echo "[tool.ruff]" >> pyproject.toml
echo "select = [\"I\"]" >> pyproject.toml
echo "line-length = 88" >> pyproject.toml
echo "[tool.ruff.isort]" >> pyproject.toml
echo "combine-as-imports = true" >> pyproject.toml
echo "force-wrap-aliases = true" >> pyproject.toml

```

## Installation

This project uses [`UV`](https://docs.astral.sh/uv/pip/compile/#locking-requirements) and `poetry` for dependency management. To set up the environment and install dependencies:

To set up, run:

```bash
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
```

To run the program:

```bash
uv run src/main.py
```

To run the tests:

```bash
uv run pytest tests
```

## Package Management

To add a new package, run:

```bash
uv add <package_name>
```

To remove a package, run:

```bash
uv remove <package_name>
```

### Setup

```bash
./setup.sh
```

### Debugging

Just import pdb and call `pdb.set_trace()` in the test you want to debug.

- `c` to continue execution
- `n` to go to the next line
- `s` to step into a function
- `l` to list the code around the current line
- `p` to print the value of a variable
- `q` to quit the debugger
  `import pprint; pprint.pprint(locals())` to print all the variables in the current scope

make sure to run the tests with the pdb flag: `pytest tests/test_dws_search.py --pdb`

### Poetry config setting

to avoid annoying package import errors make sure you set the
`pythonpath` in the `pytest.ini` file to use the `src` directory from the `pyproject.toml` file.

```ini
[tool.pytest.ini_options]
pythonpath = "src"
```

### Recommended VS Code Settings

```
{
    "[python]": {
      "editor.formatOnSave": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff",
    "ruff.lint.args": ["--config=pyproject.toml"],
     "editor.codeActionsOnSave": {
       "source.organizeImports": "explicit",
       "source.fixAll": "explicit"
     },
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic",
     "[json]": {
     "editor.wordWrap": "wordWrapColumn",
     "editor.wordWrapColumn": 80
   },

   }
```
