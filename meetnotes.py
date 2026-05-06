#!/usr/bin/env python3
"""
meetnotes — Transcribe and summarize meeting recordings locally.

Usage: meetnotes <file.webm|file.wav> [options]

Options:
    --model           mlx-whisper model (default: mlx-community/whisper-large-v3-mlx)
    --language        Language hint for Whisper, e.g. 'en', 'ru' (default: auto-detect)
    --verbose         Show Whisper segment-by-segment progress during transcription
    --lm-studio-url   LM Studio base URL (default: http://localhost:1234)
    --lm-model        LM Studio model ID (default: auto-detect from server)
    --context-size    LM Studio context length in tokens (uses native /api/v1/chat)
    --max-output-tokens Maximum summary length in generated tokens (default: 16384)
    --timeout         LM Studio request timeout in seconds (default: 120)
    --output          Output .md file path (default: <input_name>.md)
    --prompt          Custom summarization prompt file (overrides default)
    --prompt-dir      Directory containing default prompt files (default: ./prompts)
    --prompt-language Default prompt language: en, ru, de, fr, es (default: en)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    class _FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
        def log(self, *args, **kwargs): print(*args)
    console = _FallbackConsole()

def info(msg):  console.print(f"[bold cyan]→[/bold cyan] {msg}" if USE_RICH else f"→ {msg}")
def ok(msg):    console.print(f"[bold green]✓[/bold green] {msg}" if USE_RICH else f"✓ {msg}")
def err(msg):   console.print(f"[bold red]✗[/bold red] {msg}" if USE_RICH else f"✗ {msg}"); sys.exit(1)
def warn(msg):  console.print(f"[bold yellow]⚠[/bold yellow] {msg}" if USE_RICH else f"⚠ {msg}")


def ensure_wav(input_path: Path) -> Path:
    suffix = input_path.suffix.lower()

    if suffix == ".wav":
        ok(f"Input is already WAV: {input_path.name}")
        return input_path

    if suffix == ".webm":
        wav_path = input_path.with_suffix(".wav")
        info(f"Converting {input_path.name} → {wav_path.name} …")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-ar", "16000",          # 16 kHz — Whisper's native rate
                    "-ac", "1",              # mono
                    "-sample_fmt", "s16",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            err("ffmpeg not found. Install it with: brew install ffmpeg")
        except subprocess.CalledProcessError as e:
            err(f"ffmpeg failed:\n{e.stderr.decode()}")
        ok(f"Audio extracted → {wav_path.name}")
        return wav_path

    err(f"Unsupported file type '{suffix}'. Provide a .webm or .wav file.")


def transcribe(wav_path: Path, model: str, language: str | None, verbose: bool = False) -> str:
    info(f"Loading Whisper model: {model}")
    info("Transcribing … (first run downloads the model, ~3 GB)")
    if verbose:
        info("Verbose mode: Whisper will print each segment as it is decoded.")

    try:
        import mlx_whisper
    except ImportError:
        err(
            "mlx_whisper not installed in this environment.\n"
            "  Run: pip install mlx-whisper"
        )

    kwargs = dict(path_or_hf_repo=model, verbose=verbose)
    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(str(wav_path), **kwargs)
    text = result.get("text", "").strip()

    if not text:
        err("Whisper returned an empty transcript. Check the audio file.")

    ok(f"Transcription complete ({len(text):,} characters)")
    return text


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_LANGUAGE = "en"
SUPPORTED_PROMPT_LANGUAGES = ("en", "ru", "de", "fr", "es")
DEFAULT_MAX_OUTPUT_TOKENS = 16384


def default_prompt_path(prompt_dir: Path, language: str) -> Path:
    return prompt_dir / f"default_{language}.txt"


def load_prompt_template(
    custom_prompt: str | None,
    prompt_dir: str,
    prompt_language: str,
) -> str:
    if custom_prompt:
        prompt_path = Path(custom_prompt)
        if not prompt_path.exists():
            err(f"Prompt file not found: {prompt_path}")
        info(f"Using custom prompt: {prompt_path}")
    else:
        resolved_prompt_dir = Path(prompt_dir)
        if not resolved_prompt_dir.is_absolute():
            resolved_prompt_dir = SCRIPT_DIR / resolved_prompt_dir
        prompt_path = default_prompt_path(resolved_prompt_dir, prompt_language)
        if not prompt_path.exists():
            err(
                f"Default prompt not found: {prompt_path}\n"
                f"  Available default languages: {', '.join(SUPPORTED_PROMPT_LANGUAGES)}"
            )
        info(f"Using default {prompt_language} prompt: {prompt_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    if "{transcript}" not in prompt_template:
        warn("Prompt does not contain {transcript} — the transcript won't be included!")
    return prompt_template

def get_model_id(base_url: str) -> str:
    """Auto-detect the first loaded model from LM Studio."""
    import urllib.request, urllib.error
    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except urllib.error.URLError:
        pass
    return "local-model"


def extract_native_chat_content(data: dict) -> str:
    messages = [
        item.get("content", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
    ]
    return "\n\n".join(message for message in messages if message).strip()


def summarize_with_openai_compat(
    prompt: str,
    base_url: str,
    model_id: str,
    max_output_tokens: int,
    timeout: int,
) -> str:
    import urllib.request, urllib.error

    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_output_tokens,
        "stream": False,
    }).encode()

    info(f"Sending transcript to LM Studio ({base_url}) …")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        err(
            f"Could not reach LM Studio at {base_url}.\n"
            f"  Make sure LM Studio is running and a model is loaded.\n"
            f"  Error: {e}"
        )

    try:
        summary = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        err(f"Unexpected response from LM Studio:\n{json.dumps(data, indent=2)}")

    if not summary:
        err(f"LM Studio returned an empty summary:\n{json.dumps(data, indent=2)}")
    return summary


def summarize_with_native_chat(
    prompt: str,
    base_url: str,
    model_id: str,
    context_size: int,
    max_output_tokens: int,
    timeout: int,
) -> str:
    import urllib.request, urllib.error

    payload = json.dumps({
        "model": model_id,
        "input": prompt,
        "temperature": 0.3,
        "max_output_tokens": max_output_tokens,
        "context_length": context_size,
        "stream": False,
        "store": False,
    }).encode()

    info(f"Sending transcript to LM Studio ({base_url}/api/v1/chat, context: {context_size}) …")
    req = urllib.request.Request(
        f"{base_url}/api/v1/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        err(
            f"Could not reach LM Studio native chat API at {base_url}.\n"
            f"  Make sure LM Studio 0.4.0+ is running and a model is loaded.\n"
            f"  Error: {e}"
        )

    summary = extract_native_chat_content(data)
    if not summary:
        err(f"LM Studio returned an empty summary:\n{json.dumps(data, indent=2)}")
    return summary


def summarize(
    transcript: str,
    base_url: str,
    model_id: str,
    prompt_template: str,
    context_size: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: int = 120,
) -> str:
    if not model_id:
        info("Auto-detecting model from LM Studio …")
        model_id = get_model_id(base_url)
        info(f"Using model: {model_id}")

    prompt = prompt_template.replace("{transcript}", transcript)
    if context_size:
        summary = summarize_with_native_chat(
            prompt,
            base_url,
            model_id,
            context_size,
            max_output_tokens,
            timeout,
        )
    else:
        summary = summarize_with_openai_compat(
            prompt,
            base_url,
            model_id,
            max_output_tokens,
            timeout,
        )

    ok("Summarization complete")
    return summary


def save_markdown(
    output_path: Path,
    input_path: Path,
    transcript: str,
    summary: str,
) -> None:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Meeting Notes — {input_path.stem}

_Generated on {now}_

---

{summary}

---

## Full Transcript

{transcript}
"""
    output_path.write_text(md, encoding="utf-8")
    ok(f"Saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Transcribe & summarize meeting recordings using mlx-whisper + LM Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("file", help="Path to .webm or .wav recording")
    p.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-mlx",
        help="mlx-whisper HuggingFace model repo (default: whisper-large-v3-mlx)",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Language code for Whisper, e.g. 'en', 'ru' (default: auto-detect)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show Whisper segment-by-segment output during transcription (default: off)",
    )
    p.add_argument(
        "--lm-studio-url",
        default="http://localhost:1234",
        help="LM Studio server base URL (default: http://localhost:1234)",
    )
    p.add_argument(
        "--lm-model",
        default="",
        help="LM Studio model ID. Auto-detected if omitted.",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="LM Studio context length in tokens. Uses native /api/v1/chat when set.",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum summary length in generated tokens (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="LM Studio request timeout in seconds (default: 120)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .md path (default: same name as input)",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Path to a .txt file with a custom summarization prompt. "
             "Use {transcript} as placeholder.",
    )
    p.add_argument(
        "--prompt-dir",
        default=str(DEFAULT_PROMPT_DIR.relative_to(SCRIPT_DIR)),
        help="Directory containing default prompt files (default: ./prompts).",
    )
    p.add_argument(
        "--prompt-language",
        "--prompt-lang",
        choices=SUPPORTED_PROMPT_LANGUAGES,
        default=DEFAULT_PROMPT_LANGUAGE,
        help="Default prompt language to use: en, ru, de, fr, es (default: en).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.context_size is not None and args.context_size <= 0:
        err("--context-size must be a positive integer.")
    if args.max_output_tokens <= 0:
        err("--max-output-tokens must be a positive integer.")

    input_path = Path(args.file).resolve()
    if not input_path.exists():
        err(f"File not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".md")

    prompt_template = load_prompt_template(args.prompt, args.prompt_dir, args.prompt_language)

    if USE_RICH:
        console.print(Panel.fit(
            f"[bold]meetingnotes[/bold]\n[dim]{input_path.name}[/dim]",
            border_style="cyan",
        ))

    wav_path   = ensure_wav(input_path)
    transcript = transcribe(wav_path, args.model, args.language, args.verbose)
    summary    = summarize(
        transcript,
        args.lm_studio_url,
        args.lm_model,
        prompt_template,
        args.context_size,
        args.max_output_tokens,
        args.timeout,
    )
    save_markdown(output_path, input_path, transcript, summary)

    if USE_RICH:
        console.print(Panel.fit(
            f"[bold green]All done![/bold green]\n[dim]{output_path}[/dim]",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
