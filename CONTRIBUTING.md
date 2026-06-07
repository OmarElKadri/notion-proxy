# Contributing

Thanks for your interest in this project. This repository is provided for **educational purposes only** — see the [README](README.md#disclaimer).

## Before you start

- Do **not** commit secrets: `input_curl.txt`, `notion_config.json`, `.env`, session cookies, or captured request bodies.
- Keep changes focused. Prefer small, reviewable pull requests over large refactors.
- Match existing style in the files you touch (naming, formatting, prompt template conventions).

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # runtime deps + pytest
```

Run the proxy locally:

```bash
python3 -m notion_proxy
```

Run the tests:

```bash
pytest
```

## Branch workflow

1. Fork the repository (or create a branch if you have write access).
2. Create a feature branch from `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/short-description
   ```

3. Make your changes and test manually against a local Notion session.
4. Push and open a pull request against `main`.

Branch naming examples: `feat/add-health-metrics`, `fix/repair-prompt-escaping`, `docs/readme-clarification`.

## Commit messages

Use clear, imperative subject lines. One logical change per commit when possible.

**Format**

```
<type>: <short summary in imperative mood>

Optional body explaining why, not just what.
```

**Types**

| Type | Use for |
|------|---------|
| `feat` | New behavior or capability |
| `fix` | Bug fixes |
| `docs` | README, CONTRIBUTING, comments only |
| `refactor` | Code changes without behavior change |
| `chore` | Tooling, gitignore, housekeeping |

**Examples**

```
feat: load repair prompt from prompts folder

docs: document model codenames in README

fix: skip repair pass when response is already a tool call
```

## Pull request checklist

- [ ] No secrets or personal Notion session data in the diff
- [ ] README / CONTRIBUTING updated if behavior or setup changed
- [ ] Prompt template changes documented in README when placeholders change
- [ ] You have run the proxy locally and verified the change

## What to contribute

Good candidates:

- Documentation improvements
- Clearer error messages
- Safer defaults
- Prompt template tuning (with explanation in the PR)

Please avoid:

- PRs that embed live credentials or workspace-specific config
- Changes whose primary goal is bypassing Notion or third-party terms of service
- Large unrelated drive-by refactors

## Questions

Open an issue describing the problem or proposal before large changes so approach can be agreed on first.
