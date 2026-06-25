# notion-proxy

A local proxy that exposes Notion AI as an [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) endpoint. Point Claude Code (or any Anthropic-compatible client) at `http://127.0.0.1:8787` and your prompts are forwarded to Notion's `runInferenceTranscript` workflow.

## Disclaimer

**This repository is for educational purposes only.**

It exists to demonstrate how HTTP proxying, API adaptation, and prompt templating work. You are responsible for complying with [Notion's Terms of Service](https://www.notion.com/terms) and any applicable laws in your jurisdiction. The authors do not encourage misuse, credential sharing, or circumventing service limits. Use at your own risk.

> ## ⚠️ IMPORTANT — Pick the right Notion AI model
>
> | Backend | Internal codename | Recommendation |
> |---------|-------------------|----------------|
> | **GPT** | `opal-quince-medium` | **Recommended** — emits tool calls reliably, no identity conflicts |
> | Claude | `ambrosia-tart-high` | Works via a simulation-based prompt (the proxy uses a separate prompt template for Claude that frames tool calls as sample output generation). May occasionally include preamble text alongside tool calls, but the proxy parses the tool blocks correctly. |
>
> The proxy automatically selects the right prompt template based on the `model` field in `notion_config.json` — no manual configuration needed. The model alias is captured from your curl and written to the top-level `model` field. To switch backends, edit that field (or re-run the generator with `--model`). The body template's config step uses the `{{MODEL}}` placeholder, which is substituted at request time.

## Prerequisites

- Python 3.10+
- A logged-in Notion account with access to Notion AI

## Setup

### 1. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Capture a curl request from Notion

You need one real browser request to extract your session cookies, headers, and request body shape.

1. Open [Notion](https://www.notion.com/) in your browser and log in.
2. Start a **new AI conversation** (open Notion AI and begin a fresh chat).
3. Open **Developer Tools** → **Network** tab.
4. Send any short message in the chat so traffic appears in the network log.
5. Filter or scroll until you find a request named **`runInferenceTranscript`**.
6. Right-click that request → **Copy** → **Copy as cURL (bash)**.
7. Paste the entire curl command into a file named `input_curl.txt` in this repo root.

`input_curl.txt` is gitignored — it contains your live session cookie and must not be committed.

### 3. Generate `notion_config.json`

With the virtual environment still active:

```bash
python3 generate_notion_config.py input_curl.txt -o notion_config.json
```

To bake in a specific backend at generation time:

```bash
python3 generate_notion_config.py input_curl.txt -o notion_config.json --model ambrosia-tart-high
```

On Windows you can use `python` instead of `python3` if that is what your install exposes.

The script:

- Parses the curl command (URL, headers, cookies, JSON body)
- Strips noisy tracing headers
- Rewrites volatile IDs and timestamps into placeholders (`{{PROMPT}}`, `{{NOW}}`, etc.)
- Parameterizes the model alias as `{{MODEL}}` and records the default in the top-level `model` field
- Writes a reusable `notion_config.json`

`notion_config.json` is also gitignored. It contains your session cookie — treat it like a password.

See `notion_config.example.json` for the expected shape if you need to inspect or hand-edit fields.

### 4. Run the proxy

```bash
python3 -m notion_proxy
```

Defaults: `127.0.0.1:8787`. Override with `--host`, `--port`, or `--config`:

```bash
python3 -m notion_proxy --port 8787 --config notion_config.json
```

Health check: `GET http://127.0.0.1:8787/health`

### Prompt templates

Prompt text lives in `prompts/` — edit these files to change proxy behavior without touching Python code.

| File | When it runs | Placeholders |
|------|--------------|--------------|
| `prompts/proxy_to_notion.txt` | Every request (GPT backend) | `{{ERROR_NUDGE}}`, `{{SYSTEM_PROMPT}}`, `{{TOOLS_SUMMARY}}`, `{{CONVERSATION_TRANSCRIPT}}` |
| `prompts/proxy_to_notion_claude.txt` | Every request (Claude backend) | Same placeholders — uses a simulation framing so Claude emits tool calls despite Notion's identity anchoring |
| `prompts/repair_self_reference.txt` | Only when Notion replies as "Notion AI" instead of continuing the agent conversation | `{{AVAILABLE_TOOLS}}`, `{{ORIGINAL_PROMPT}}`, `{{BAD_RESPONSE}}` |

The repair template is used by `repair_self_referential_response()` in `notion_proxy/repair.py`: after each Notion response, if the text matches self-referential patterns (e.g. "I'm Notion AI", "I can't access local files") and isn't already a valid tool call, the proxy sends a second Notion request with the repair prompt to rewrite the answer.

## Running the tests

The pure logic (tool-call parsing, validation, stream parsing, body rendering, response building) is covered by a pytest suite:

```bash
pip install -r requirements-dev.txt
pytest
```

## Using with Claude Code

Configure your Anthropic-compatible client to use the local base URL:

| Setting | Value |
|---------|-------|
| Base URL | `http://127.0.0.1:8787` |
| API key | any non-empty string (not validated locally) |
| Model | any string (passed through; Notion ignores it) |

The proxy accepts `POST /v1/messages` with streaming and non-streaming bodies, maps the payload into a Notion transcript prompt, and translates the NDJSON response back into Anthropic SSE events.

## Refreshing your config

Notion session cookies expire. When requests start failing with auth errors:

1. Repeat the curl capture steps above into a fresh `input_curl.txt`.
2. Re-run `generate_notion_config.py` to overwrite `notion_config.json`.
3. Restart the proxy.

## Project layout

| File | Purpose |
|------|---------|
| `notion_proxy/` | The proxy package (run with `python -m notion_proxy`) |
| `notion_proxy/app.py` | FastAPI app factory and `/v1/messages` route |
| `notion_proxy/prompt.py` | Prompt-template loading and request-body rendering |
| `notion_proxy/tools/` | Tool-call parsing (`parse.py`) and validation (`validate.py`) |
| `notion_proxy/notion/` | Notion HTTP client (`client.py`) and NDJSON stream parser (`stream.py`) |
| `notion_proxy/responses.py` | Anthropic JSON + SSE response builders |
| `notion_proxy/repair.py` | Self-referential-reply repair pass |
| `generate_notion_config.py` | Converts a saved curl command into `notion_config.json` |
| `extract_claude_context.py` | Helper to inspect Claude Code request payloads |
| `tests/` | Pytest suite (`pytest`) |
| `notion_config.example.json` | Annotated config template (safe to commit) |
| `input_curl.txt` | Your captured curl (local only, gitignored) |
| `notion_config.json` | Generated runtime config (local only, gitignored) |
| `prompts/proxy_to_notion.txt` | Main prompt template (GPT backend) |
| `prompts/proxy_to_notion_claude.txt` | Claude backend prompt (simulation framing) |
| `prompts/repair_self_reference.txt` | Fallback prompt when Notion gives a self-referential reply |

## Security notes

- Never commit `input_curl.txt` or `notion_config.json` — both contain session tokens.
- The proxy binds to `127.0.0.1` by default; do not expose it to the public internet without additional auth.
- Debug logging (`last_response.txt`, `log.txt`) may contain prompt content; both are gitignored.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch workflow, commit message conventions, and pull request guidelines.

Quick flow:

1. Branch from `main`: `git checkout -b feat/your-change`
2. Commit with a clear message: `feat: describe what changed`
3. Push and open a PR against `main`
4. Confirm no secrets are in the diff

## License

MIT — see [LICENSE](LICENSE).

---

**Made By OTYAK**
