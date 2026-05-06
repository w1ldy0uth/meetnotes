# meetnotes

Transcribe and summarize your team meetings locally

## Usage

Use the helper script to run meetnotes inside the local virtual environment:

```bash
./meetnotes meeting.webm
```

The script creates `venv` if needed, installs `requirements.txt`, and forwards all arguments to `meetnotes.py`.

Set LM Studio context length for large meetings:

```bash
./meetnotes meeting.webm --context-size 32768
```

When `--context-size` is set, meetnotes uses LM Studio's native `/api/v1/chat` endpoint because the OpenAI-compatible chat endpoint does not support per-request context length.

Set the maximum summary length separately:

```bash
./meetnotes meeting.webm --context-size 32768 --max-output-tokens 4096
```

`--context-size` controls how much input and output the model can fit in memory. `--max-output-tokens` controls how long the generated summary is allowed to be.

## Prompt selection

By default, meetnotes uses `prompts/default_en.txt`.

Use another bundled prompt language:

```bash
./meetnotes meeting.webm --prompt-language ru
```

Bundled default languages: `en`, `ru`, `de`, `fr`, `es`.

Use a different defaults directory:

```bash
./meetnotes meeting.webm --prompt-dir ./my-prompts --prompt-language de
```

Use a fully custom prompt file:

```bash
./meetnotes meeting.webm --prompt ./custom_prompt.txt
```

Prompt files should contain `{transcript}` where the transcript should be inserted.
