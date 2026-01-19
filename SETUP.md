# SETUP

> Getting and running this project.

## 01: Set Up Machine (Once Per Machine)

Follow the detailed instructions at:
[**01. Set Up Your Machine**](https://denisecase.github.io/pro-analytics-02/01-set-up-machine/)

## 02: Set Up Project (Once Per Project)

1. Sign in to GitHub, open this repository in your browser, and click **Copy this template** to get a copy in **YOURACCOUNT**.
2. Enable GitHub Pages.
3. Open a **machine terminal** in your `Repos` folder and clone your new repo.
4. Change directory into the repo, open the project in VS Code, and install recommended extensions.
5. Set up a project Python environment (managed by `uv`) and align VS Code with it.

Use the instructions above to get it ALL set up correctly.
Most people open a terminal on their machine (not VS Code), open in their Repos folder and run:

```shell
git clone https://github.com/YOURACCOUNT/toy-gpt-train

cd toy-gpt-train
code .
```

When VS Code opens, accept the Extension Recommendations (click **`Install All`** or similar when asked).

Use VS Code menu option `Terminal` / `New Terminal` to open a **VS Code terminal** in the root project folder.
Run the following commands, one at a time, hitting ENTER after each:

```shell
uv self update
uv python pin 3.14
uv cache clean
uv sync --extra dev --extra docs --upgrade
```

If asked: "We noticed a new environment has been created. Do you want to select it for the workspace folder?" Click **"Yes"**.

Install and run pre-commit checks (repeat git `add` and `commit` twice as needed):

```shell
uvx pre-commit install
git add -A
uvx pre-commit run --all-files
```

More detailed instructions are available at:
[**02. Set Up Your Project**](https://denisecase.github.io/pro-analytics-02/02-set-up-project/)

## 03: Daily Workflow (Working With Python Project Code)

Commands are provided below to:

1. Git pull
2. Run and check the Python files
3. Build and serve docs
4. Save progress with Git add-commit-push
5. Update project files

VS Code should have only this project (toy-gpt-train) open.
Use VS Code menu option `Terminal` / `New Terminal` and run the following commands:

```shell
git pull
```

In the same VS Code terminal, run the Python source files:

```shell
uv run python src/toy_gpt_train_llm/a_tokenizer.py
uv run python src/toy_gpt_train_llm/b_vocab.py
uv run python src/toy_gpt_train_llm/c_model.py
uv run python src/toy_gpt_train_llm/d_train.py
uv run python src/toy_gpt_train_llm/e_infer.py
```

If a command fails, verify:

- Only this project is open in VS Code.
- The terminal is open in the project root folder.
- The `uv sync --extra dev --extra docs --upgrade` command completed successfully.

Hint: if you run `ls` in the terminal, you should see files including `pyproject.toml`, `README.md`, and `uv.lock`.

Run Python checks and tests (as available):

```shell
uv run ruff format .
uv run ruff check . --fix
uv run -- python -m pytest

uv run pyright
uv run bandit -c pyproject.toml -r src
uv run validate-pyproject pyproject.toml
```

Build and serve docs (hit **CTRL+c** in the VS Code terminal to quit serving):

```shell
uv run mkdocs build --strict
uv run mkdocs serve
```

While editing project code and docs, repeat the commands above to run files, check them, and rebuild docs as needed.

Save progress frequently (some tools may make changes; you may need to **re-run git `add` and `commit`** to ensure everything gets committed before pushing):

```shell
git add -A
git commit -m "update"
git push -u origin main
```

Follow the detailed instructions at:
[**03. Daily Workflow**](https://denisecase.github.io/pro-analytics-02/03-daily-workflow/)

## Resources

- [Pro-Analytics-02](https://denisecase.github.io/pro-analytics-02/) - guide to professional Python
- [ANNOTATIONS.md](./ANNOTATIONS.md) - REQ/WHY/OBS annotations used
- [SE_MANIFEST.toml](./SE_MANIFEST.toml) - project intent, scope, and role

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
