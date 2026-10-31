#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import codecs
import atexit
import pprint
import shlex
import shutil
import socket
import platform
import subprocess
import webbrowser

import urllib.parse
import urllib.request

from datetime import datetime
from xml.sax.saxutils import escape
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Generator, Dict, Callable

try:
    import tkinter as tk
    from tkinter import ttk, font, messagebox

except (ModuleNotFoundError, ImportError):
    print(
        "Your Python installation does not include the Tk library. \n"
        "Please refer to https://github.com/Botspot/ollama-gui?tab=readme-ov-file#troubleshooting")
    sys.exit(0)

__version__ = "1.2.1"

HF_HOST = "https://huggingface.co"
HF_SEARCH_LIMIT = 30

# Mirrors kappai's src/arch.c registry, any other architecture fails to load
SUPPORTED_ARCHS = ("llama", "gemma4", "gemma4_moe", "qwen35", "qwen3vl", "glm-dsa", "glm5-next", "lfm2")
# Model families from kappai's README support matrix, searched to build the supported model list
CATALOG_SEARCHES = ("Llama-3.2", "gemma-4", "Qwen3.5", "Qwen3.6", "Qwen3.8", "Qwen3-VL",
                    "LFM2.5", "GLM-5")
CATALOG_PER_SEARCH = 20
# Curated picks for people who just want to chat, smallest first, sized by their Q4_K_M file
RECOMMENDED = [
    {"id": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "file": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
     "size": 0.68 * 2**30, "params": 1.2e9, "arch": "lfm2", "note": "tiny and fast"},
    {"id": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_K_M.gguf",
     "size": 1.19 * 2**30, "params": 1.9e9, "arch": "qwen35", "note": "small, thinks before answering"},
    {"id": "unsloth/Llama-3.2-3B-Instruct-GGUF", "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "size": 1.88 * 2**30, "params": 3.2e9, "arch": "llama", "note": "small all-rounder"},
    {"id": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_K_M.gguf",
     "size": 2.55 * 2**30, "params": 4.2e9, "arch": "qwen35", "note": "balanced, thinks before answering"},
    {"id": "unsloth/gemma-4-E4B-it-GGUF", "file": "gemma-4-E4B-it-Q4_K_M.gguf",
     "size": 4.64 * 2**30, "params": 7.5e9, "arch": "gemma4", "note": "balanced all-rounder"},
    {"id": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf",
     "size": 5.29 * 2**30, "params": 9.0e9, "arch": "qwen35", "note": "smart but slower"},
    {"id": "unsloth/gemma-4-12b-it-GGUF", "file": "gemma-4-12b-it-Q4_K_M.gguf",
     "size": 6.63 * 2**30, "params": 11.9e9, "arch": "gemma4", "note": "smart but slower"},
    {"id": "unsloth/gemma-4-26B-A4B-it-GGUF", "file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
     "size": 15.78 * 2**30, "params": 25.2e9, "arch": "gemma4", "note": "most capable, needs lots of RAM"},
]
CATALOG_MAX_AGE = 24 * 3600
CATALOG_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "kappai-gui", "catalog.json")

# Flat look with a 2px grey border that turns blue on focus, used by all classic tk widgets
OUTLINE = {
    "relief": "flat",
    "highlightthickness": 2,
    "highlightbackground": "#CCCCCC",
    "highlightcolor": "#0078D7",
}

# kappai defaults to the model's full trained context, whose KV cache rarely fits in RAM
CTX_SIZE = 8192
# Mesa's V3D driver crashes on kappai's OpenGL backend, the CPU backend is reliable everywhere
DEFAULT_SERVER_OPTIONS = "--device cpu0"
# Conversations are user data rather than cache, one JSON file per chat named by its start time
CHATS_DIR = os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
    "kappai-gui", "chats")
CHAT_ID_FORMAT = "%Y-%m-%d_%H-%M-%S"
NEW_CHAT_LABEL = "New chat"
TEMPORARY_CHAT_LABEL = "Temporary chat (not saved)"
GUI_SETTINGS = os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
    "kappai-gui", "settings.json")

MIB = 1024 * 1024
# IQ2_XXS, the smallest quantization kappai loads, used to judge repos by parameter count
MIN_BITS_PER_WEIGHT = 2.06
# KV cache and OS headroom on top of kappai's own estimate of weights * 1.05 + 256 MiB
MEMORY_HEADROOM = 768 * MIB

# Mirrors kappai's src/tui/hf.c so downloads land where the kappai TUI expects them
MODEL_SUFFIXES = (".gguf", ".safetensors")
NON_CHAT_TYPES = ("image", "audio", "embedding")
UNSUPPORTED_QUANTS = ("Q1", "Q2_K", "Q8_K", "IQ1_", "IQ2_XS", "TQ", "MXFP", "NVFP")
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
PIPELINE_TYPES = {
    "text-generation": "language", "text2text-generation": "language",
    "image-text-to-text": "language", "question-answering": "language",
    "summarization": "language", "translation": "language",
    "text-to-image": "image", "image-to-image": "image", "text-to-video": "image",
    "image-to-video": "image", "unconditional-image-generation": "image",
    "automatic-speech-recognition": "audio", "text-to-speech": "audio",
    "text-to-audio": "audio", "audio-to-audio": "audio",
    "feature-extraction": "embedding", "sentence-similarity": "embedding",
    "reinforcement-learning": "decision", "robotics": "decision",
}


class DownloadCancelled(Exception):
    pass


def _kappai_settings() -> tuple:
    """Read models_dir and custom models from kappai's TUI settings file"""
    base = os.environ.get("XDG_CONFIG_HOME", "")
    if not base.startswith("/") or len(base) < 2:
        base = os.path.expanduser("~/.config")
    models_dir, custom = "", []
    try:
        with open(os.path.join(base, "sys64", "kappai", "tui.conf")) as f:
            for line in f:
                key, sep, value = line.partition("=")
                key, value = key.strip().lower(), value.strip()
                if not sep:
                    continue
                if key == "models_dir":
                    models_dir = value
                elif key == "custom_model":
                    custom.append(value)
    except OSError:
        pass
    return models_dir or os.path.expanduser("~/models"), custom


def _find_server() -> str:
    return (os.environ.get("KAPPAI_SERVER")
            or shutil.which("kappai-server")
            or os.path.expanduser("~/kappai/build/kappai-server"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def _detect_quant(path: str) -> str:
    stem = re.sub(r"\.(gguf|safetensors)$", "", path, flags=re.IGNORECASE)
    quant = "?"
    for token in re.split(r"[-./]", stem):
        upper = token.upper()
        if upper in ("BF16", "F16", "F32") or re.match(r"(Q\d|IQ\d|MXFP)", upper):
            quant = upper
    return quant


def _sanitize_dir(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "-", name.lower()) or "unknown"


def _hf_json(path: str):
    with urllib.request.urlopen(HF_HOST + path, timeout=30) as response:
        return json.load(response)


def _hf_search(query: str, limit: int) -> List[dict]:
    repos = _hf_json("/api/models?" + urllib.parse.urlencode([
        ("search", query), ("filter", "gguf"), ("sort", "downloads"), ("direction", -1),
        ("limit", limit), ("expand[]", "gguf"), ("expand[]", "downloads"),
        ("expand[]", "pipeline_tag"),
    ]))
    results = []
    for repo in repos:
        gguf = repo.get("gguf") or {}
        results.append({
            "id": repo["id"],
            "downloads": repo.get("downloads") or 0,
            "arch": gguf.get("architecture") or "",
            "params": gguf.get("total") or 0,
            "pipeline": repo.get("pipeline_tag") or "",
        })
    return results


def _repo_supported(repo: dict) -> bool:
    return (repo["arch"] in SUPPORTED_ARCHS
            and PIPELINE_TYPES.get(repo["pipeline"], "language") == "language")


def _total_memory() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _fits_memory(size: float, memory: int) -> bool:
    return not memory or not size or size * 1.05 + 256 * MIB + MEMORY_HEADROOM <= memory


def _prefer_color_emoji():
    """
    Tk draws each character with the first font in fontconfig's fallback list that has it,
    so emoji come out in whatever monochrome or differently styled font happens to be listed
    before the color emoji font. Point fontconfig at an app-only config that puts the color
    emoji font right after the text font. It can't go first, emoji fonts also contain digits.
    Must run before Tk starts.
    """
    system_conf = "/etc/fonts/fonts.conf"
    if (platform.system() != "Linux" or "FONTCONFIG_FILE" in os.environ
            or not os.path.exists(system_conf) or not shutil.which("fc-match")):
        return

    def fc_match(pattern: str, fmt: str) -> str:
        return subprocess.run(["fc-match", "-f", fmt, pattern], capture_output=True,
                              text=True, timeout=10).stdout.strip()

    try:
        emoji = fc_match("emoji", "%{family[0]}")
        text = fc_match("sans-serif", "%{family[0]}")
        if not text or fc_match(emoji, "%{color}") != "True":
            return
        path = os.path.join(os.path.dirname(CATALOG_CACHE), "fonts.conf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(
                '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
                f'  <include ignore_missing="yes">{system_conf}</include>\n'
                '  <match target="pattern">\n'
                f'    <test qual="first" name="family"><string>{escape(text)}</string></test>\n'
                '    <edit name="family" mode="append" binding="strong">'
                f'<string>{escape(emoji)}</string></edit>\n'
                '  </match>\n</fontconfig>\n'
            )
        os.environ["FONTCONFIG_FILE"] = path
    except (OSError, subprocess.SubprocessError):
        pass


def _grab_when_viewable(window: tk.Toplevel):
    """grab_set fails until the window manager has mapped a new window, so retry until it has"""
    if not window.winfo_exists():
        return
    try:
        window.grab_set()
    except tk.TclError:
        window.after(20, lambda: _grab_when_viewable(window))


def _system_check(root: tk.Tk) -> Optional[str]:
    """
    Detected some system and software compatibility issues,
    and returned the information in the form of a string to alert the user

    :param root: Tk instance
    :return: None or message string
    """

    def _version_tuple(v):
        """A lazy way to avoid importing third-party libraries"""
        filled = []
        for point in v.split("."):
            filled.append(point.zfill(8))
        return tuple(filled)

    # Tcl and macOS issue: https://github.com/python/cpython/issues/110218
    if platform.system().lower() == "darwin":
        version = platform.mac_ver()[0]
        if version and 14 <= float(version) < 15:
            tcl_version = root.tk.call("info", "patchlevel")
            if _version_tuple(tcl_version) <= _version_tuple("8.6.12"):
                return (
                    "Warning: Tkinter Responsiveness Issue Detected\n\n"
                    "You may experience unresponsive GUI elements when "
                    "your cursor is inside the window during startup. "
                    "This is a known issue with Tcl/Tk versions 8.6.12 "
                    "and older on macOS Sonoma.\n\nTo resolve this:\n"
                    "Update to Python 3.11.7+ or 3.12+\n"
                    "Or install Tcl/Tk 8.6.13 or newer separately\n\n"
                    "Temporary workaround: Move your cursor out of "
                    "the window and back in if elements become unresponsive.\n\n"
                    "For more information, visit: https://github.com/python/cpython/issues/110218"
                )


MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)[\s#]*$")
MD_RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
MD_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
MD_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
MD_NUMBER_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
MD_INLINE_RE = re.compile(
    r"(?P<tick>`+)(?P<code>.+?)(?P=tick)"
    r"|\*\*\*(?P<bold_italic>\S(?:.*?\S)?)\*\*\*"
    r"|\*\*(?P<bold>\S(?:.*?\S)?)\*\*"
    r"|(?<!\w)__(?P<bold_>\S(?:.*?\S)?)__(?!\w)"
    r"|(?<![\w*])\*(?P<italic>[^\s*](?:.*?[^\s*])?)\*(?![\w*])"
    r"|(?<!\w)_(?P<italic_>[^\s_](?:.*?[^\s_])?)_(?!\w)"
    r"|~~(?P<strike>\S(?:.*?\S)?)~~"
    r"|\[(?P<link>[^\]\n]+)\]\((?P<href>https?://[^)\s]+)\)"
)
# Inline tags a match produces, keyed by the regex group holding the visible text
MD_INLINE_TAGS = {"code": "code", "bold_italic": "bold_italic", "bold": "bold",
                  "bold_": "bold", "italic": "italic", "italic_": "italic",
                  "strike": "strike", "link": "link"}
MD_LIST_INDENT = 20
MD_MAX_LEVEL = 3


def _markdown_inline(text: str, tags: tuple) -> List[tuple]:
    segments, pos = [], 0
    for match in MD_INLINE_RE.finditer(text):
        if match.start() > pos:
            segments.append((text[pos:match.start()], tags))
        group = next(g for g in MD_INLINE_TAGS if match.group(g) is not None)
        extra = (MD_INLINE_TAGS[group],)
        if group == "link":
            extra += ("href:" + match.group("href"),)
        segments.append((match.group(group), tags + extra))
        pos = match.end()
    if pos < len(text):
        segments.append((text[pos:], tags))
    return segments


def _markdown_segments(raw: str) -> List[tuple]:
    """
    The markdown a Text widget can show with tags: headings, emphasis, inline and fenced code,
    lists, quotes, rules and links. Tables and other layouts are left as plain text.
    Returns (text, tags) pairs. Unclosed fences and emphasis stay readable mid-stream.
    """
    segments = []
    in_code = False
    lines = raw.split("\n")
    for n, line in enumerate(lines):
        end = "\n" if n < len(lines) - 1 else ""
        if MD_FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code:
            segments.append((line + end, ("code_block",)))
            continue

        heading = MD_HEADING_RE.match(line)
        quote = MD_QUOTE_RE.match(line)
        bullet = MD_BULLET_RE.match(line)
        number = MD_NUMBER_RE.match(line)
        if heading:
            tags = (f"h{min(len(heading.group(1)), 3)}",)
            segments += _markdown_inline(heading.group(2), tags)
        elif MD_RULE_RE.match(line):
            tags = ("rule",)
            segments.append(("─" * 24, tags))
        elif quote:
            tags = ("quote",)
            segments += _markdown_inline(quote.group(1), tags)
        elif bullet or number:
            match = bullet or number
            level = min(len(match.group(1).expandtabs(4)) // 2, MD_MAX_LEVEL)
            tags = (f"{'bullet' if bullet else 'number'}{level}",)
            segments.append(("• " if bullet else match.group(2) + " ", tags))
            segments += _markdown_inline(match.group(match.lastindex), tags)
        else:
            tags = ()
            segments += _markdown_inline(line, tags)
        if end:
            segments.append((end, tags))
    return segments


class MarkdownStyles:
    """Fonts and tag options shared by every markdown bubble"""
    # Later entries win when tags overlap, Text gives later-configured tags priority
    FONT_PRIORITY = ("strike", "italic", "bold", "bold_italic", "code", "code_block",
                     "h3", "h2", "h1")

    def __init__(self, base: font.Font):
        family, size = base.actual("family"), base.actual("size")
        mono = font.nametofont("TkFixedFont").actual("family")
        self.base = base
        self.fonts = {
            "strike": font.Font(family=family, size=size, overstrike=True),
            "italic": font.Font(family=family, size=size, slant="italic"),
            "bold": font.Font(family=family, size=size, weight="bold"),
            "bold_italic": font.Font(family=family, size=size, weight="bold", slant="italic"),
            "code": font.Font(family=mono, size=size - 1),
            "h3": font.Font(family=family, size=size + 2, weight="bold"),
            "h2": font.Font(family=family, size=size + 4, weight="bold"),
            "h1": font.Font(family=family, size=size + 6, weight="bold"),
        }
        self.fonts["code_block"] = self.fonts["code"]
        self.margins = {"quote": 12, "code_block": 6}
        bullet, number = base.measure("• "), base.measure("00. ")
        self.hanging = {}
        for level in range(MD_MAX_LEVEL + 1):
            self.margins[f"bullet{level}"] = self.margins[f"number{level}"] = level * MD_LIST_INDENT
            self.hanging[f"bullet{level}"] = level * MD_LIST_INDENT + bullet
            self.hanging[f"number{level}"] = level * MD_LIST_INDENT + number

    def configure(self, text: tk.Text, code_background: str):
        for tag, margin in self.margins.items():
            text.tag_configure(tag, lmargin1=margin, lmargin2=self.hanging.get(tag, margin))
        text.tag_configure("quote", foreground="#666666")
        text.tag_configure("rule", foreground="#999999")
        text.tag_configure("code_block", background=code_background, wrap=tk.CHAR)
        text.tag_configure("link", foreground="#0b5cad", underline=True)
        for name in self.FONT_PRIORITY:
            text.tag_configure(name, font=self.fonts[name])
        text.tag_configure("code", background=code_background)
        for name in ("h1", "h2", "h3"):
            text.tag_configure(name, spacing1=4, spacing3=2)

    def font_for(self, tags: tuple) -> font.Font:
        for name in reversed(self.FONT_PRIORITY):
            if name in tags:
                return self.fonts[name]
        return self.base

    def margin_for(self, tags: tuple) -> int:
        return max((self.margins.get(tag, 0) for tag in tags), default=0)


class ChatBubble(tk.Frame):
    """
    Chat bubble that shrinks to fit its text up to max_width like a wrapping Label, but built
    on a read-only Text widget so its text can be selected and copied. The frame is sized in
    pixels and the Text fills it, because a Text can only be sized in characters and lines.
    With markdown styles the raw text is rendered, at most every RENDER_MS while streaming.
    """
    PAD = 8
    RENDER_MS = 100

    def __init__(self, master: tk.Widget, bubble_font: font.Font, max_width: int,
                 background: str, foreground: str, select_background: str,
                 on_resize: Callable[[], None], markdown: Optional[MarkdownStyles] = None):
        super().__init__(master, background=background, borderwidth=0, highlightthickness=0)
        self.pack_propagate(False)
        self.font = bubble_font
        self.max_width = max_width
        self.on_resize = on_resize
        self.markdown = markdown
        self.raw = ""
        self.render_pending = False
        self.longest = 0
        self.width = 0

        self.text = tk.Text(
            self,
            font=bubble_font,
            background=background,
            foreground=foreground,
            selectbackground=select_background,
            inactiveselectbackground=select_background,
            selectforeground=foreground,
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            padx=self.PAD,
            pady=self.PAD,
            insertwidth=0,
            cursor="xterm",
            state=tk.DISABLED,
        )
        self.text.pack(fill="both", expand=True)
        # The bubble always shows all of its text, so never let it scroll internally, e.g.
        # when a drag-selection leaves the bubble or the keyboard moves through it
        self.text.config(yscrollcommand=lambda first, last:
                         float(first) > 0 and self.text.yview_moveto(0))
        self.text.bind("<B1-Leave>", lambda e: "break")
        # Disabled Text widgets don't take focus when clicked, which Ctrl+C needs
        self.text.bind("<Button-1>", lambda e: self.text.focus_set(), add="+")
        # The height can only be measured once the new width has actually been applied
        self.text.bind("<Configure>", lambda e: self._fit_height(), add="+")
        if markdown:
            markdown.configure(self.text, code_background="#d9d9d9")
            self.text.tag_bind("link", "<Button-1>", self._open_link)
            self.text.tag_bind("link", "<Enter>", lambda e: self.text.config(cursor="hand2"))
            self.text.tag_bind("link", "<Leave>", lambda e: self.text.config(cursor="xterm"))
        self._fit_width()

    def get_text(self) -> str:
        """The raw text, markdown included"""
        return self.raw

    def selected_text(self) -> str:
        ranges = self.text.tag_ranges("sel")
        return self.text.get(*ranges) if ranges else ""

    def set_text(self, value: str):
        self.raw = value
        self._render()

    def append(self, value: str):
        self.raw += value
        if not self.render_pending:
            self.render_pending = True
            self.after(self.RENDER_MS, self._render)

    def _render(self):
        self.render_pending = False
        if not self.winfo_exists():
            return
        segments = _markdown_segments(self.raw) if self.markdown else [(self.raw, ())]
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        flat = [item for text, tags in segments if text for item in (text, tags)]
        if flat:
            self.text.insert(tk.END, *flat)
        self.text.config(state=tk.DISABLED)
        self.longest = self._measure(segments)
        self._fit_width()

    def _measure(self, segments: List[tuple]) -> int:
        """Widest line in pixels, each piece measured in the font its tags give it"""
        longest = line = 0
        margin = None
        for text, tags in segments:
            text_font = self.markdown.font_for(tags) if self.markdown else self.font
            for i, part in enumerate(text.split("\n")):
                if i:
                    longest = max(longest, line + (margin or 0))
                    line, margin = 0, None
                if part:
                    if margin is None:
                        margin = self.markdown.margin_for(tags) if self.markdown else 0
                    line += text_font.measure(part)
        return max(longest, line + (margin or 0))

    def _open_link(self, event: tk.Event):
        for tag in self.text.tag_names(f"@{event.x},{event.y}"):
            if tag.startswith("href:"):
                webbrowser.open(tag[5:])
                return "break"

    def set_max_width(self, max_width: int):
        if max_width != self.max_width:
            self.max_width = max_width
            self._fit_width()

    def _fit_width(self):
        # A couple of spare pixels so a line of exactly the measured width doesn't wrap
        width = max(min(self.longest + 2 * self.PAD + 2, self.max_width), 2 * self.PAD + 10)
        if width != self.width:
            self.width = width
            self.config(width=width)
        self._fit_height()

    def _fit_height(self):
        if abs(self.text.winfo_width() - self.width) > 1:
            return
        content = int(self.text.tk.call(self.text._w, "count", "-update", "-ypixels",
                                        "1.0", "end"))
        height = content + 2 * self.PAD
        if height != int(self.cget("height")):
            self.config(height=height)
            self.on_resize()


class KappaiInterface:
    chat_box: tk.Text
    user_input: tk.Text
    models_dir_input: ttk.Entry
    status_label: ttk.Label
    progress: ttk.Progressbar
    stop_button: ttk.Button
    send_button: ttk.Button
    refresh_button: ttk.Button
    search_button: ttk.Button
    download_button: ttk.Button
    cancel_download_button: ttk.Button
    download_progress: ttk.Progressbar
    download_progress_label: ttk.Label
    chat_select: ttk.Combobox
    delete_button: ttk.Button
    model_select: ttk.Combobox
    log_textbox: tk.Text
    models_list: tk.Listbox
    repos_list: tk.Listbox
    files_list: tk.Listbox

    def __init__(self, root: tk.Tk):
        self.root: tk.Tk = root
        self.models_dir, self.custom_models = _kappai_settings()
        self.chat_history: List[dict] = []
        self.label_widgets: List[ChatBubble] = []
        self.generating: bool = False
        self.autoscroll: bool = True
        self.scroll_pending: bool = False
        self.default_font: str = font.nametofont("TkTextFont").actual()["family"]

        self.installed: List[dict] = []
        self.chat_models: Dict[str, str] = {}
        self.repo: Optional[dict] = None
        self.catalog: List[dict] = []
        self.repo_results: List[dict] = []
        self.repo_view: tuple = ([], "")
        self.memory: int = _total_memory()
        self.show_all = tk.BooleanVar(value=False)
        self.recommended = tk.BooleanVar(value=True)

        self.server: Optional[subprocess.Popen] = None
        self.server_model: str = ""
        self.api_url: str = ""
        self.server_args: List[str] = []
        self.gui_settings: dict = self.load_gui_settings()
        self.server_options = tk.StringVar(
            value=self.gui_settings.get("server_options", DEFAULT_SERVER_OPTIONS))
        self.server_options.trace_add(
            "write", lambda *_: self.save_gui_settings(server_options=self.server_options.get()))
        self.server_lock = Lock()
        self.download_cancel = Event()
        self.downloading: bool = False
        atexit.register(self.stop_server)

        self.chat_id: Optional[str] = None
        self.chat_temporary: bool = False
        self.chat_labels: Dict[str, str] = {}

        self.layout = LayoutManager(self)
        self.layout.init_layout()

        self.root.after(200, self.check_system)
        self.root.after_idle(self.resume_last_chat)
        self.refresh_models()

    def copy_text(self, text: str):
        if text:
            self.chat_box.clipboard_clear()
            self.chat_box.clipboard_append(text)

    def copy_all(self):
        self.copy_text(pprint.pformat(self.chat_history))

    @staticmethod
    def open_homepage():
        webbrowser.open("https://github.com/Botspot/ollama-gui")

    def show_help(self):
        info = ("Project: Kappai GUI\n"
                f"Version: {__version__}\n"
                "Author: Botspot, based on Ollama GUI by chyok\n"
                "Github: https://github.com/Botspot/ollama-gui\n"
                "Engine: https://github.com/System64fumo/kappai\n\n"
                "<Enter>: send\n"
                "<Shift+Enter>: new line\n"
                "<Right click message>: edit or copy\n"
                "Editing your message replaces everything below it with a new response\n")
        messagebox.showinfo("About", info, parent=self.root)

    def check_system(self):
        message = _system_check(self.root)
        if message is not None:
            messagebox.showwarning("Warning", message, parent=self.root)

    def append_text_to_chat(self,
                            text: str,
                            *args,
                            use_label: bool = False):
        if use_label:
            self.label_widgets[-1].append(text)
        else:
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.insert(tk.END, text, *args)
            self.chat_box.config(state=tk.DISABLED)
        self.request_scroll()

    def request_scroll(self):
        if self.autoscroll and not self.scroll_pending:
            self.scroll_pending = True
            self.root.after_idle(self.stick_to_bottom)

    def stick_to_bottom(self):
        self.scroll_pending = False
        if self.autoscroll:
            # Let the text widget lay out the grown bubble first, or the last line gets cut off
            self.chat_box.update_idletasks()
            self.chat_box.yview_moveto(1.0)

    def update_autoscroll(self):
        """Keep following new output only while the user is scrolled to the very bottom"""
        self.autoscroll = self.chat_box.yview()[1] >= 0.9995

    def on_chat_scrollbar(self, *args):
        self.chat_box.yview(*args)
        self.update_autoscroll()

    def on_chat_wheel(self, event: tk.Event):
        # Button-4/5 on X11, MouseWheel with a signed delta on Windows and macOS
        up = event.num == 4 or (event.num != 5 and event.delta > 0)
        self.chat_box.yview_scroll(-50 if up else 50, "pixels")
        self.update_autoscroll()
        return "break"

    def bind_chat_wheel(self, widget: tk.Widget):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self.on_chat_wheel)

    def append_log_to_inner_textbox(self,
                                    message: Optional[str] = None,
                                    clear: bool = False):
        if self.log_textbox.winfo_exists():
            self.log_textbox.config(state=tk.NORMAL)
            if clear:
                self.log_textbox.delete(1.0, tk.END)
            elif message:
                self.log_textbox.insert(tk.END, message + "\n")
            self.log_textbox.config(state=tk.DISABLED)
            self.log_textbox.see(tk.END)

    def resize_inner_text_widget(self, event: tk.Event):
        max_width = int(event.widget.winfo_width() * 0.7)
        for bubble in self.label_widgets:
            bubble.set_max_width(max_width)

    def clear_chat_selection(self, _=None):
        """Clicking between bubbles clears the selection instead of selecting chat_box text"""
        for bubble in self.label_widgets:
            bubble.text.tag_remove("sel", "1.0", tk.END)
        return "break"

    def show_error(self, text):
        self.model_select.set(text)
        self.model_select.config(foreground="red")
        self.model_select["values"] = []
        self.send_button.state(["disabled"])

    def show_process_bar(self):
        self.progress.grid(row=0, column=0, sticky="nsew")
        self.stop_button.grid(row=0, column=1, padx=20)
        self.progress.start(5)

    def hide_process_bar(self):
        self.progress.stop()
        self.stop_button.grid_remove()
        self.progress.grid_remove()

    def handle_key_press(self, event: tk.Event):
        if event.keysym == "Return":
            if event.state & 0x1 == 0x1:  # Shift key is pressed
                self.user_input.insert("end", "\n")
            elif "disabled" not in self.send_button.state():
                self.on_send_button(event)
            return "break"

    def set_status(self, text: str, color: str = "gray"):
        self.status_label.config(text=text, foreground=color)

    def refresh_models(self):
        self.update_models_dir()
        self.model_select.config(foreground="black")
        self.model_select.set("Waiting...")
        self.send_button.state(["disabled"])
        self.refresh_button.state(["disabled"])
        Thread(target=self.update_model_select, daemon=True).start()

    def update_models_dir(self):
        self.models_dir = os.path.expanduser(self.models_dir_input.get().strip())

    def update_model_select(self):
        try:
            self.scan_models()
            models = list(self.chat_models)
            self.model_select["values"] = models
            self.layout.fit_model_select()
            if self.server_model in self.chat_models.values():
                self.model_select.set(next(k for k, v in self.chat_models.items()
                                           if v == self.server_model))
                self.send_button.state(["!disabled"])
            elif models:
                self.model_select.set(models[0])
                self.send_button.state(["!disabled"])
            else:
                self.show_error("You need download a model!")
        except Exception:  # noqa
            self.show_error("Error! Please check the models folder.")
        finally:
            self.refresh_button.state(["!disabled"])

    def update_model_list(self):
        if self.models_list.winfo_exists():
            self.models_list.delete(0, tk.END)
            try:
                self.scan_models()
                for model in self.installed:
                    label = f"{model['rel']}  ({_format_bytes(model['size'])}"
                    if model["parts"] > 1:
                        label += f", {model['parts']} parts"
                    self.models_list.insert(tk.END, label + ")")
            except Exception as e:
                self.append_log_to_inner_textbox(f"Error reading models folder: {e}")

    def on_model_selected(self, _=None):
        path = self.chat_models.get(self.model_select.get())
        if path:
            Thread(target=self.load_model, daemon=True, args=(path,)).start()

    def load_model(self, path: str):
        try:
            self.ensure_server(path)
        except Exception as e:
            self.set_status(str(e), "red")

    @staticmethod
    def load_gui_settings() -> dict:
        try:
            with open(GUI_SETTINGS) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_gui_settings(self, **changes):
        self.gui_settings.update(changes)
        try:
            os.makedirs(os.path.dirname(GUI_SETTINGS), exist_ok=True)
            with open(GUI_SETTINGS, "w") as f:
                json.dump(self.gui_settings, f, indent=2)
        except OSError as e:
            print(f"Could not save {GUI_SETTINGS}: {e}", file=sys.stderr)

    def ensure_server(self, path: str):
        with self.server_lock:
            try:
                args = shlex.split(self.server_options.get())
            except ValueError as e:
                raise RuntimeError(f"Invalid server options: {e}")
            if (self.server and self.server.poll() is None and self.server_model == path
                    and self.server_args == args):
                return
            self.stop_server()
            exe = _find_server()
            if not os.access(exe, os.X_OK):
                raise RuntimeError(f"kappai-server not found at {exe} "
                                   "(add it to PATH or set KAPPAI_SERVER)")
            port = _free_port()
            self.set_status(f"Loading {os.path.basename(path)}...", "orange")
            # User options come after our defaults to override them, host and port last
            # because the GUI needs them to reach the server
            self.server = subprocess.Popen(
                [exe, "-m", path, "-c", str(CTX_SIZE), *args,
                 "--host", "127.0.0.1", "--port", str(port)]
            )
            self.server_model = path
            self.server_args = args
            self.api_url = f"http://127.0.0.1:{port}"
            while self.server.poll() is None:
                try:
                    with urllib.request.urlopen(self.api_url + "/health", timeout=2):
                        self.set_status(f"Ready: {os.path.basename(path)}", "green")
                        return
                except OSError:
                    time.sleep(0.5)
            code = self.server.returncode
            self.server, self.server_model = None, ""
            raise RuntimeError(f"kappai-server exited with code {code}, see terminal output")

    def stop_server(self):
        server, self.server, self.server_model = self.server, None, ""
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        try:
            self.set_status("No model loaded")
        except (tk.TclError, RuntimeError):
            pass

    def scan_models(self):
        """Find models the same way kappai's model library does (4 levels deep, shards merged)"""
        found = []
        root = self.models_dir
        for dirpath, dirnames, filenames in os.walk(root):
            depth = 0 if dirpath == root else os.path.relpath(dirpath, root).count(os.sep) + 1
            dirnames[:] = [d for d in dirnames if not d.startswith(".")] if depth < 4 else []
            for name in filenames:
                if name.startswith(".") or not name.lower().endswith(MODEL_SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                found.append((os.path.relpath(path, root), path, os.path.getsize(path)))
        found.sort(key=lambda m: m[0].lower())

        installed = []
        for rel, path, size in found:
            key = SHARD_RE.sub("", rel) if SHARD_RE.search(rel) else None
            if key and installed and installed[-1]["key"] == key:
                installed[-1]["size"] += size
                installed[-1]["parts"] += 1
                continue
            installed.append({"rel": rel, "path": path, "size": size, "parts": 1,
                              "key": key, "custom": False})
        for path in self.custom_models:
            if os.path.isfile(path):
                installed.append({"rel": path, "path": path, "size": os.path.getsize(path),
                                  "parts": 1, "key": None, "custom": True})

        self.installed = installed
        self.chat_models = {
            m["rel"]: m["path"] for m in installed
            if m["rel"].lower().endswith(".gguf")
            and "mmproj" not in os.path.basename(m["rel"]).lower()
            and m["rel"].split(os.sep, 1)[0] not in NON_CHAT_TYPES
        }

    def on_send_button(self, _=None):
        message = self.user_input.get("1.0", "end-1c")
        if message:
            self.layout.create_inner_label(on_right_side=True)
            self.append_text_to_chat(f"{message}", use_label=True)
            self.append_text_to_chat(f"\n\n")
            self.user_input.delete("1.0", "end")
            self.chat_history.append({"role": "user", "content": message})
            self.save_chat()
            self.start_response()

    def start_response(self):
        self.generating = True
        self.autoscroll = True
        Thread(target=self.generate_ai_response, daemon=True).start()

    def edit_message(self, bubble: ChatBubble, text: str):
        idx = self.label_widgets.index(bubble)
        if idx >= len(self.chat_history):
            return
        self.chat_history[idx]["content"] = text
        bubble.set_text(text)
        if self.chat_history[idx]["role"] != "user":
            self.save_chat()
            return

        # Everything below an edited user message answered the old version, drop it and re-ask
        for later in self.label_widgets[idx + 1:]:
            later.destroy()
        del self.label_widgets[idx + 1:]
        del self.chat_history[idx + 1:]
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete(f"{self.chat_box.index(str(bubble))} + 1c", tk.END)
        self.chat_box.config(state=tk.DISABLED)
        self.append_text_to_chat("\n\n")
        self.save_chat()
        self.start_response()

    def generate_ai_response(self):
        self.show_process_bar()
        self.send_button.state(["disabled"])
        self.refresh_button.state(["disabled"])

        ai_message = ""
        bubble_created = False
        model = self.model_select.get()
        try:
            self.ensure_server(self.chat_models[model])
            self.append_text_to_chat(f"{model}\n", ("Bold",))
            self.layout.create_inner_label()
            bubble_created = True
            for i in self.fetch_chat_stream_result():
                self.append_text_to_chat(f"{i}", use_label=True)
                ai_message += i
            self.chat_history.append({"role": "assistant", "content": ai_message, "model": model})
            self.append_text_to_chat("\n\n")
        except Exception as e:
            # Keep chat_history in step with the bubbles, editing relies on matching indexes
            if bubble_created:
                self.chat_history.append(
                    {"role": "assistant", "content": ai_message, "model": model})
            self.append_text_to_chat(f"\nAI error: {e}\n\n", ("Error",))
        finally:
            self.save_chat()
            self.generating = False
            if self.server_model:
                self.set_status(f"Ready: {os.path.basename(self.server_model)}", "green")
            self.hide_process_bar()
            self.send_button.state(["!disabled"])
            self.refresh_button.state(["!disabled"])
            self.stop_button.state(["!disabled"])

    def fetch_chat_stream_result(self) -> Generator:
        request = urllib.request.Request(
            self.api_url + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.model_select.get(),
                    "messages": [{"role": m["role"], "content": m["content"]}
                                 for m in self.chat_history],
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # kappai streams raw token bytes, so a multi-byte character can be split across
        # chunks. Keep undecodable bytes as surrogates and reassemble them incrementally.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with urllib.request.urlopen(request) as resp:
            for line in resp:
                if "disabled" in self.stop_button.state():  # stop
                    break
                line = line.decode("utf-8", errors="surrogateescape").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                for choice in json.loads(payload).get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        self.set_status("Thinking...", "orange")
                    if delta.get("content"):
                        text = decoder.decode(
                            delta["content"].encode("utf-8", errors="surrogateescape"))
                        if text:
                            yield text
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail

    def delete_model(self, index: Optional[int]):
        self.append_log_to_inner_textbox(clear=True)
        if index is None or index >= len(self.installed):
            return
        model = self.installed[index]
        if model["custom"]:
            self.append_log_to_inner_textbox(
                "This is a custom model path, remove it in kappai's settings instead.")
            return

        directory = os.path.dirname(model["path"])
        if model["key"]:
            prefix = os.path.basename(model["key"])
            paths = [os.path.join(directory, f) for f in os.listdir(directory)
                     if SHARD_RE.search(f) and SHARD_RE.sub("", f) == prefix]
        else:
            paths = [model["path"]]

        try:
            if self.server_model == model["path"]:
                self.stop_server()
            for path in paths:
                os.remove(path)
            root = os.path.abspath(self.models_dir)
            while os.path.abspath(directory) != root and not os.listdir(directory):
                os.rmdir(directory)
                directory = os.path.dirname(directory)
            self.append_log_to_inner_textbox(f"Deleted {model['rel']}")
        except Exception as e:
            self.append_log_to_inner_textbox(f"Failed to delete model: {e}")
        finally:
            self.update_model_list()
            self.update_model_select()

    def load_catalog(self):
        """Show the cached supported model list straight away, refresh it from Hugging Face when stale"""
        self.repo_view = (self.catalog, "supported models")
        cached_time = 0
        try:
            with open(CATALOG_CACHE) as f:
                data = json.load(f)
            self.catalog, cached_time = data["repos"], data["time"]
        except (OSError, ValueError, KeyError):
            pass
        if self.catalog:
            self.show_repos(self.catalog, "supported models")
        if time.time() - cached_time < CATALOG_MAX_AGE:
            return

        self.append_log_to_inner_textbox("Updating the supported model list from Hugging Face...")
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                found = {}
                for repos in pool.map(lambda q: _hf_search(q, CATALOG_PER_SEARCH),
                                      CATALOG_SEARCHES):
                    found.update((r["id"], r) for r in repos if _repo_supported(r))
            self.catalog = sorted(found.values(), key=lambda r: -r["downloads"])
            os.makedirs(os.path.dirname(CATALOG_CACHE), exist_ok=True)
            with open(CATALOG_CACHE, "w") as f:
                json.dump({"time": time.time(), "repos": self.catalog}, f)
        except Exception as e:
            self.append_log_to_inner_textbox(f"Could not update the supported model list: {e}")
            return
        if self.repo_view[1] == "supported models":
            self.show_repos(self.catalog, "supported models")

    def show_repos(self, repos: List[dict], what: str):
        if not self.repos_list.winfo_exists():
            return
        self.repo_view = (repos, what)
        shown = [r for r in repos if self.show_all.get() or _fits_memory(
            r.get("size") or r["params"] * MIN_BITS_PER_WEIGHT / 8, self.memory)]
        self.repo_results = shown
        self.repos_list.delete(0, tk.END)
        for repo in shown:
            label = repo["id"]
            if repo["params"]:
                label += f"  ({repo['params'] / 1e9:.1f}B, {repo['arch']})"
            if repo.get("note"):
                label += f"  {repo['note']}"
            self.repos_list.insert(tk.END, label)
        log = f"{len(shown)} {what}"
        if len(shown) < len(repos):
            log += (f", {len(repos) - len(shown)} too large for "
                    f"{_format_bytes(self.memory)} of RAM hidden")
        self.append_log_to_inner_textbox(log)

    def show_files(self) -> int:
        if self.repo is None or not self.files_list.winfo_exists():
            return 0
        all_files = self.repo["all_files"]
        self.repo["files"] = [f for f in all_files if self.show_all.get()
                              or _fits_memory(f["total"], self.memory)]
        recommended = next((r["file"] for r in RECOMMENDED if r["id"] == self.repo["id"]), None)
        self.files_list.delete(0, tk.END)
        for i, f in enumerate(self.repo["files"]):
            label = f"{f['quant']:<8} {_format_bytes(f['total']):>10}  {f['name']}"
            if f["name"] == recommended:
                label += "  (recommended)"
            self.files_list.insert(tk.END, label)
            if f["name"] == recommended:
                self.files_list.selection_set(i)
                self.files_list.see(i)
        return len(all_files) - len(self.repo["files"])

    def on_show_all_toggled(self):
        self.show_repos(*self.repo_view)
        self.show_files()

    def on_recommended_toggled(self):
        self.append_log_to_inner_textbox(clear=True)
        if self.recommended.get():
            self.show_repos(RECOMMENDED, "recommended models")
        else:
            Thread(target=self.load_catalog, daemon=True).start()

    def search_repos(self, query: str):
        if not query:
            self.on_recommended_toggled()
            return
        self.append_log_to_inner_textbox(clear=True)
        self.recommended.set(False)
        if re.fullmatch(r"[\w.-]+/[\w.-]+", query) and ".." not in query:
            self.open_repo(query)
            return

        self.search_button.state(["disabled"])
        self.append_log_to_inner_textbox(f"Searching Hugging Face for '{query}'...")
        try:
            repos = _hf_search(query, HF_SEARCH_LIMIT)
            supported = [r for r in repos if not r["arch"] or _repo_supported(r)]
            self.files_list.delete(0, tk.END)
            self.show_repos(supported, f"results for '{query}'")
            if len(supported) < len(repos):
                self.append_log_to_inner_textbox(
                    f"{len(repos) - len(supported)} hidden, not a chat model kappai supports")
        except Exception as e:
            self.append_log_to_inner_textbox(f"Search failed: {e}")
        finally:
            if self.search_button.winfo_exists():
                self.search_button.state(["!disabled"])

    def open_repo(self, repo: str):
        self.append_log_to_inner_textbox(clear=True)
        self.append_log_to_inner_textbox(f"Loading {repo}...")
        self.repo = None
        self.files_list.delete(0, tk.END)
        try:
            info = _hf_json(f"/api/models/{repo}?blobs=true")
            gguf = info.get("gguf") or {}
            files, unsupported = [], 0
            for sibling in info.get("siblings") or []:
                name = sibling.get("rfilename", "")
                # mmproj files are vision projectors, not chat models
                if (not name.lower().endswith(".gguf") or name.startswith("/")
                        or ".." in name or "mmproj" in name.lower()):
                    continue
                quant = _detect_quant(name)
                if quant.startswith(UNSUPPORTED_QUANTS):
                    unsupported += 1
                    continue
                size = (sibling.get("lfs") or sibling).get("size") or 0
                key = SHARD_RE.sub("", name) if SHARD_RE.search(name) else name
                files.append({"name": name, "size": size, "quant": quant, "key": key})
        except Exception as e:
            self.append_log_to_inner_textbox(f"Failed to load repository: {e}")
            return

        totals: Dict[str, int] = {}
        for f in files:
            totals[f["key"]] = totals.get(f["key"], 0) + f["size"]
        for f in files:
            f["total"] = totals[f["key"]]

        arch = gguf.get("architecture", "")
        self.repo = {
            "id": repo,
            "all_files": files,
            "type": PIPELINE_TYPES.get(info.get("pipeline_tag", ""), "other"),
            "arch": _sanitize_dir(arch),
        }
        too_large = self.show_files()
        log = f"{repo}: {len(files)} GGUF files"
        if unsupported:
            log += f", {unsupported} hidden (quantization not supported by kappai)"
        if too_large:
            log += f", {too_large} hidden (too large for {_format_bytes(self.memory)} of RAM)"
        self.append_log_to_inner_textbox(log)
        if arch and arch not in SUPPORTED_ARCHS:
            self.append_log_to_inner_textbox(
                f"Warning: kappai does not support the '{arch}' architecture")

    def download_model(self, index: Optional[int]):
        self.append_log_to_inner_textbox(clear=True)
        if self.repo is None or index is None or index >= len(self.repo["files"]):
            return

        chosen = self.repo["files"][index]
        group = [f for f in self.repo["all_files"] if f["key"] == chosen["key"]]

        self.download_cancel.clear()
        self.downloading = True
        self.update_download_buttons()
        target = os.path.join(self.models_dir, self.repo["type"], self.repo["arch"])
        total = sum(f["size"] for f in group)
        done = 0
        self.show_download_progress(0, total, 0)
        try:
            for i, f in enumerate(group, 1):
                self.append_log_to_inner_textbox(
                    f"[{i}/{len(group)}] {f['name']} ({_format_bytes(f['size'])})")
                self.download_file(self.repo["id"], f, os.path.join(target, f["name"]),
                                   done, total)
                done += f["size"]
            self.append_log_to_inner_textbox(
                f"Download complete: {os.path.relpath(target, self.models_dir)}"
                f"/{group[0]['name']}")
        except DownloadCancelled:
            self.append_log_to_inner_textbox(
                "Download cancelled, the partial file will resume next time.")
        except Exception as e:
            self.append_log_to_inner_textbox(f"Failed to download model: {e}")
        finally:
            self.downloading = False
            self.update_download_buttons()
            self.show_download_progress(0, 0, 0)
            self.update_model_list()
            self.update_model_select()

    def show_download_progress(self, done: int, total: int, speed: float):
        if not self.download_progress.winfo_exists():
            return
        self.download_progress["value"] = done * 100 / total if total else 0
        if not done and not speed:
            text = ""
        elif total:
            text = (f"{done * 100 // total}%   {_format_bytes(done)} / {_format_bytes(total)}"
                    f"   {_format_bytes(speed)}/s")
        else:
            text = f"{_format_bytes(done)}   {_format_bytes(speed)}/s"
        self.download_progress_label.config(text=text)

    def update_download_buttons(self):
        if self.download_button.winfo_exists():
            self.download_button.state(["disabled" if self.downloading else "!disabled"])
            self.cancel_download_button.state(["!disabled" if self.downloading else "disabled"])

    def download_file(self, repo: str, f: dict, dest: str, done_before: int, total: int):
        """Resumable download into dest.part, same layout as kappai's TUI downloader"""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        size = f["size"]
        if size and os.path.exists(dest) and os.path.getsize(dest) == size:
            self.append_log_to_inner_textbox("Already downloaded")
            return

        part = dest + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if not (size and have >= size):
            url = f"{HF_HOST}/{repo}/resolve/main/{urllib.parse.quote(f['name'])}"
            headers = {"Range": f"bytes={have}-"} if have else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60) as response:
                if response.status != 206:
                    have = 0
                started, start_have, last_update = time.time(), have, 0.0
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        if self.download_cancel.is_set():
                            raise DownloadCancelled()
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last_update > 0.5:
                            last_update = now
                            speed = (have - start_have) / max(now - started, 0.5)
                            self.show_download_progress(done_before + have, total, speed)
        os.replace(part, dest)

    def busy(self, action: str) -> bool:
        if self.generating:
            messagebox.showinfo(
                "Busy", f"Wait for the response to finish, or press Stop, before {action}.",
                parent=self.root)
        return self.generating

    def clear_chat_view(self):
        for i in self.label_widgets:
            i.destroy()
        self.label_widgets.clear()
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete(1.0, tk.END)
        self.chat_box.config(state=tk.DISABLED)

    def clear_chat(self):
        """Empty the current chat, a saved chat left empty is deleted from disk"""
        if self.busy("clearing"):
            return
        self.clear_chat_view()
        self.chat_history.clear()
        self.save_chat()

    @staticmethod
    def chat_path(chat_id: str) -> str:
        return os.path.join(CHATS_DIR, chat_id + ".json")

    @staticmethod
    def chat_title(chat_id: str, seconds: bool = False) -> str:
        try:
            started = datetime.strptime(chat_id, CHAT_ID_FORMAT)
        except ValueError:
            return chat_id
        return started.strftime("%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M")

    def save_chat(self):
        """Write the conversation to disk after every message, temporary chats never are"""
        if self.chat_temporary:
            return
        if not self.chat_history:
            if self.chat_id:
                try:
                    os.remove(self.chat_path(self.chat_id))
                except OSError:
                    pass
                self.chat_id = None
                self.refresh_chat_list()
            return

        new = self.chat_id is None
        if new:
            self.chat_id = datetime.now().strftime(CHAT_ID_FORMAT)
        path = self.chat_path(self.chat_id)
        try:
            os.makedirs(CHATS_DIR, exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump({"version": 1, "messages": self.chat_history}, f,
                          ensure_ascii=False, indent=1)
            os.replace(path + ".tmp", path)
        except OSError as e:
            self.set_status(f"Could not save chat: {e}", "red")
            return
        if new:
            self.save_gui_settings(last_chat=self.chat_id)
            self.refresh_chat_list()

    def refresh_chat_list(self):
        """Saved chats newest first, with the current chat shown in the selector"""
        try:
            ids = [name[:-5] for name in os.listdir(CHATS_DIR) if name.endswith(".json")]
        except OSError:
            ids = []
        ids.sort(key=lambda i: os.path.getmtime(self.chat_path(i)), reverse=True)
        self.chat_labels = {}
        for chat_id in ids:
            title = self.chat_title(chat_id)
            if title in self.chat_labels:
                title = self.chat_title(chat_id, seconds=True)
            self.chat_labels[title] = chat_id

        self.chat_select["values"] = list(self.chat_labels)
        if self.chat_temporary:
            self.chat_select.set(TEMPORARY_CHAT_LABEL)
        elif self.chat_id is None:
            self.chat_select.set(NEW_CHAT_LABEL)
        else:
            self.chat_select.set(next((label for label, i in self.chat_labels.items()
                                       if i == self.chat_id), self.chat_id))

    def render_history(self):
        self.clear_chat_view()
        for message in self.chat_history:
            if message["role"] == "user":
                self.layout.create_inner_label(on_right_side=True)
            else:
                self.append_text_to_chat(f"{message.get('model', 'assistant')}\n", ("Bold",))
                self.layout.create_inner_label()
            self.label_widgets[-1].set_text(message["content"])
            self.append_text_to_chat("\n\n")
        self.autoscroll = True
        self.request_scroll()

    def open_chat(self, chat_id: str) -> bool:
        try:
            with open(self.chat_path(chat_id)) as f:
                messages = json.load(f)["messages"]
        except (OSError, ValueError, KeyError) as e:
            self.set_status(f"Could not open chat: {e}", "red")
            return False
        self.chat_history = messages
        self.chat_id = chat_id
        self.chat_temporary = False
        self.save_gui_settings(last_chat=chat_id)
        self.render_history()
        self.refresh_chat_list()
        return True

    def resume_last_chat(self):
        last = self.gui_settings.get("last_chat")
        if not (last and os.path.exists(self.chat_path(last)) and self.open_chat(last)):
            self.refresh_chat_list()

    def new_chat(self, temporary: bool = False):
        if self.busy("starting a new chat"):
            return
        self.chat_history = []
        self.chat_id = None
        self.chat_temporary = temporary
        self.clear_chat_view()
        self.refresh_chat_list()
        self.user_input.focus_set()

    def on_chat_selected(self, _=None):
        chat_id = self.chat_labels.get(self.chat_select.get())
        if self.busy("switching chats") or not chat_id:
            self.refresh_chat_list()
            return
        self.open_chat(chat_id)


class LayoutManager:
    """
    Manages the layout and arrangement of the KappaiInterface.

    The LayoutManager is responsible for the visual organization and positioning
    of the various components within the KappaiInterface, such as the header,
    chat container, progress bar, and input fields. It handles the sizing,
    spacing, and alignment of these elements to create a cohesive and
    user-friendly layout.
    """

    def __init__(self, interface: KappaiInterface):
        self.interface: KappaiInterface = interface
        self.management_window: Optional[tk.Toplevel] = None
        self.editor_window: Optional[tk.Toplevel] = None
        self.popup: Optional[tk.Menu] = None
        self.bubble_font: Optional[font.Font] = None
        self.markdown_styles: Optional[MarkdownStyles] = None

    def init_layout(self):
        self._header_frame()
        self._chat_container_frame()
        self._processbar_frame()
        self._input_frame()
        self.interface.root.bind("<FocusOut>", self._on_focus_out, add="+")

    def show_popup(self, menu: tk.Menu, event: tk.Event):
        """tk_popup grabs input so the next click anywhere dismisses the menu, unlike post()"""
        self.close_popup()
        self.popup = menu
        menu.tk_popup(event.x_root, event.y_root)

    def close_popup(self):
        if self.popup and self.popup.winfo_exists():
            self.popup.unpost()
            self.popup.grab_release()
        self.popup = None

    def _on_focus_out(self, _):
        # FocusOut also fires when focus moves into the menu itself, so only close the menu
        # once focus has actually left the application
        def _check():
            try:
                focused = self.interface.root.focus_get()
            except KeyError:  # focus is on a Tk-internal widget, still inside the app
                return
            if focused is None:
                self.close_popup()
        self.interface.root.after(50, _check)

    def _header_frame(self):
        header_frame = ttk.Frame(self.interface.root)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        header_frame.grid_columnconfigure(3, weight=1)

        model_select = ttk.Combobox(header_frame, state="readonly", width=30)
        model_select.grid(row=0, column=0)
        model_select.bind("<<ComboboxSelected>>", self.interface.on_model_selected)

        settings_button = ttk.Button(
            header_frame, text="Settings", command=self.show_model_management_window
        )
        settings_button.grid(row=0, column=1, padx=(5, 0))

        refresh_button = ttk.Button(header_frame, text="Refresh", command=self.interface.refresh_models)
        refresh_button.grid(row=0, column=2, padx=(5, 0))

        status_label = ttk.Label(header_frame, text="No model loaded", foreground="gray")
        status_label.grid(row=0, column=3, sticky="w", padx=(10, 0))

        ttk.Label(header_frame, text="Models:").grid(row=0, column=4, padx=(10, 0))

        models_dir_input = ttk.Entry(header_frame, width=24)
        models_dir_input.grid(row=0, column=5, padx=(5, 15))
        models_dir_input.insert(0, self.interface.models_dir)

        chat_frame = ttk.Frame(header_frame)
        chat_frame.grid(row=1, column=0, columnspan=6, sticky="w", pady=(10, 0))

        chat_select = ttk.Combobox(chat_frame, state="readonly", width=30)
        chat_select.grid(row=0, column=0)
        chat_select.bind("<<ComboboxSelected>>", self.interface.on_chat_selected)

        ttk.Button(chat_frame, text="New Chat", command=self.interface.new_chat).grid(
            row=0, column=1, padx=(5, 0))
        ttk.Button(chat_frame, text="Temporary Chat",
                   command=lambda: self.interface.new_chat(temporary=True)).grid(
            row=0, column=2, padx=(5, 0))

        self.interface.chat_select = chat_select

        self.header_frame = header_frame
        header_frame.bind("<Configure>", self.fit_model_select)

        self.interface.model_select = model_select
        self.interface.refresh_button = refresh_button
        self.interface.status_label = status_label
        self.interface.models_dir_input = models_dir_input

    def fit_model_select(self, _=None):
        """Widen the model selector to fit the longest model name, as far as the window allows"""
        select = self.interface.model_select
        status = self.interface.status_label
        text_font = font.nametofont("TkTextFont")
        char_width = text_font.measure("0")
        names = list(select["values"] or ()) + [select.get()]
        # Room for the dropdown arrow and padding
        needed = max(text_font.measure(name) for name in names) + 40

        others = (self.header_frame.winfo_reqwidth() - select.winfo_reqwidth()
                  - status.winfo_reqwidth() + min(status.winfo_reqwidth(), 200))
        available = self.header_frame.winfo_width() - others
        chars = max(min(needed, available) // char_width, 30)
        if int(select.cget("width")) != chars:
            select.config(width=chars)

    def _chat_container_frame(self):
        chat_frame = ttk.Frame(self.interface.root)
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=20)
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)

        chat_box = tk.Text(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=(self.interface.default_font, 12),
            spacing1=5,
            **OUTLINE,
        )
        chat_box.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical",
                                  command=self.interface.on_chat_scrollbar)
        scrollbar.grid(row=0, column=1, sticky="ns")

        chat_box.configure(yscrollcommand=scrollbar.set)
        self.interface.bind_chat_wheel(chat_box)
        for sequence in ("<Button-1>", "<B1-Motion>", "<Double-1>", "<Triple-1>",
                         "<Shift-Button-1>", "<B1-Leave>"):
            chat_box.bind(sequence, self.interface.clear_chat_selection)

        chat_box_menu = tk.Menu(chat_box, tearoff=0)
        chat_box_menu.add_command(label="Copy All", command=self.interface.copy_all)
        chat_box_menu.add_separator()
        chat_box_menu.add_command(label="Clear Chat", command=self.interface.clear_chat)
        chat_box.bind("<Configure>", self.interface.resize_inner_text_widget)

        _right_click = (
            "<Button-2>" if platform.system().lower() == "darwin" else "<Button-3>"
        )
        chat_box.bind(_right_click, lambda e: self.show_popup(chat_box_menu, e))

        self.interface.chat_box = chat_box

    def _processbar_frame(self):
        process_frame = ttk.Frame(self.interface.root, height=28)
        process_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        progress = ttk.Progressbar(
            process_frame,
            mode="indeterminate",
            style="LoadingBar.Horizontal.TProgressbar",
        )

        stop_button = ttk.Button(
            process_frame,
            width=5,
            text="Stop",
            command=lambda: stop_button.state(["disabled"]),
        )

        self.interface.progress = progress
        self.interface.stop_button = stop_button

    def _input_frame(self):
        input_frame = ttk.Frame(self.interface.root)
        input_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_frame.grid_columnconfigure(0, weight=1)

        # Updated user_input with flat relief and a blue highlight border
        user_input = tk.Text(
            input_frame, 
            font=(self.interface.default_font, 12), 
            height=4, 
            wrap=tk.WORD,
            relief="flat",                 # Removes the 3D shaded effect
            highlightthickness=2,          # Sets the thickness of the border
            highlightbackground="#CCCCCC", # The border color when the text box is NOT focused (light gray)
            highlightcolor="#0078D7"       # The nice blue border color when the text box IS focused
        )
        user_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        user_input.bind("<Key>", self.interface.handle_key_press)

        send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.interface.on_send_button,
        )
        send_button.grid(row=0, column=1)
        send_button.state(["disabled"])

        menubar = tk.Menu(self.interface.root)
        self.interface.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Model Management", command=self.show_model_management_window)
        file_menu.add_command(label="Exit", command=self.interface.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy All", command=self.interface.copy_all)
        edit_menu.add_command(label="Clear Chat", command=self.interface.clear_chat)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Source Code", command=self.interface.open_homepage)
        help_menu.add_command(label="Help", command=self.interface.show_help)

        self.interface.user_input = user_input
        self.interface.send_button = send_button

    def show_model_management_window(self):
        self.interface.update_models_dir()

        if self.management_window and self.management_window.winfo_exists():
            self.management_window.lift()
            return

        management_window = tk.Toplevel(self.interface.root)
        management_window.title("Model Management")
        
        # Remove absolute positioning and let the compositor handle it
        management_window.geometry("560x720")
        
        # Tell the window manager this is a dialog tied to the main window
        management_window.transient(self.interface.root)
        
        # Tell the Linux WM to treat this as a dialog/utility (skips taskbar)
        if platform.system().lower() == "linux":
            management_window.attributes("-type", "dialog")
        
        # Grab focus so the user must interact with this window before returning
        _grab_when_viewable(management_window)

        management_window.grid_columnconfigure(0, weight=1)
        for row in (2, 3, 5):
            management_window.grid_rowconfigure(row, weight=1)

        frame = ttk.Frame(management_window)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)

        search_input = ttk.Entry(frame)
        search_input.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        def _selected(listbox: tk.Listbox) -> Optional[int]:
            selection = listbox.curselection()
            return selection[0] if selection else None

        def _search(_=None):
            arg = search_input.get().strip()
            Thread(target=self.interface.search_repos, daemon=True, args=(arg,)).start()

        def _open_repo(_=None):
            idx = _selected(repos_list)
            if idx is not None and idx < len(self.interface.repo_results):
                repo = self.interface.repo_results[idx]["id"]
                Thread(target=self.interface.open_repo, daemon=True, args=(repo,)).start()

        def _download():
            Thread(
                target=self.interface.download_model, daemon=True,
                args=(_selected(files_list),)
            ).start()

        def _delete():
            Thread(
                target=self.interface.delete_model, daemon=True,
                args=(_selected(models_list),)
            ).start()

        search_input.bind("<Return>", _search)
        search_button = ttk.Button(frame, text="Search", command=_search)
        search_button.grid(row=0, column=1, sticky="ew")

        tips = tk.Label(
            frame,
            text="search huggingface.co, enter owner/repo, or leave empty for supported models",
            fg="blue",
            cursor="hand2",
        )
        tips.bind("<Button-1>",
                  lambda e: webbrowser.open("https://huggingface.co/models?library=gguf"))
        tips.grid(row=1, column=0, columnspan=2, sticky="W", padx=(0, 5), pady=(5, 0))

        show_all_check = ttk.Checkbutton(
            frame,
            text=f"Show models too large for this device "
                 f"({_format_bytes(self.interface.memory)} RAM)",
            variable=self.interface.show_all,
            command=self.interface.on_show_all_toggled,
        )
        show_all_check.grid(row=2, column=0, columnspan=2, sticky="W")

        recommended_check = ttk.Checkbutton(
            frame,
            text="Recommended",
            variable=self.interface.recommended,
            command=self.interface.on_recommended_toggled,
        )
        recommended_check.grid(row=3, column=0, columnspan=2, sticky="W")

        def _list_frame(row: int, button_text: Optional[str] = None, command=None):
            list_frame = ttk.Frame(management_window)
            list_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=(0, 10))
            list_frame.grid_columnconfigure(0, weight=1)
            list_frame.grid_rowconfigure(0, weight=1)

            listbox = tk.Listbox(
                list_frame,
                **OUTLINE,
                exportselection=False,
                height=6,
            )
            listbox.grid(row=0, column=0, sticky="nsew")

            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            listbox.config(yscrollcommand=scrollbar.set)

            button = None
            if button_text:
                button = ttk.Button(list_frame, text=button_text, command=command)
                button.grid(row=0, column=2, sticky="new", padx=(5, 0))
            return listbox, button

        repos_list, _ = _list_frame(2, "Open", _open_repo)
        repos_list.bind("<Double-1>", _open_repo)
        files_list, download_button = _list_frame(3, "Download", _download)
        files_list.config(font=("TkFixedFont",))
        cancel_download_button = ttk.Button(
            download_button.master, text="Cancel", command=self.interface.download_cancel.set)
        cancel_download_button.grid(row=0, column=2, sticky="sew", padx=(5, 0))

        ttk.Style().configure("Download.Horizontal.TProgressbar", background="#0078D7")
        download_progress = ttk.Progressbar(
            download_button.master, mode="determinate", maximum=100,
            style="Download.Horizontal.TProgressbar")
        download_progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        download_progress_label = ttk.Label(download_button.master, foreground="gray")
        download_progress_label.grid(row=2, column=0, columnspan=3, sticky="w")

        ttk.Label(management_window, text="Installed models:").grid(
            row=4, column=0, sticky="w", padx=10)
        models_list, delete_button = _list_frame(5, "Delete", _delete)

        # Updated log_textbox
        log_textbox = tk.Text(
            management_window,
            height=6,
            **OUTLINE,
        )
        log_textbox.grid(row=6, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_textbox.config(state="disabled")

        self.management_window = management_window

        self.interface.log_textbox = log_textbox
        self.interface.search_button = search_button
        self.interface.download_button = download_button
        self.interface.cancel_download_button = cancel_download_button
        self.interface.download_progress = download_progress
        self.interface.download_progress_label = download_progress_label
        self.interface.update_download_buttons()
        self.interface.delete_button = delete_button
        self.interface.models_list = models_list
        self.interface.repos_list = repos_list
        self.interface.files_list = files_list

        server_frame = ttk.Frame(management_window)
        server_frame.grid(row=7, column=0, sticky="ew", padx=10, pady=(0, 10))
        server_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(server_frame, text="Server options:").grid(row=0, column=0, padx=(0, 5))
        server_options_input = ttk.Entry(server_frame, textvariable=self.interface.server_options)
        server_options_input.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        server_options_input.bind("<Return>", self.interface.on_model_selected)

        server_help_button = ttk.Button(server_frame, text="Help", command=self.show_server_help)
        server_help_button.grid(row=0, column=2)
        Thread(
            target=self.interface.update_model_list, daemon=True,
        ).start()
        self.interface.on_recommended_toggled()

    def show_server_help(self):
        exe = _find_server()
        try:
            result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=10)
            help_text = result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            help_text = f"Could not run {exe} --help: {e}"

        help_window = tk.Toplevel(self.interface.root)
        help_window.title("kappai-server options")
        help_window.geometry("720x560")
        parent = self.management_window or self.interface.root
        help_window.transient(parent)
        if platform.system().lower() == "linux":
            help_window.attributes("-type", "dialog")
        # Take over the management window's input grab, and hand it back on close
        _grab_when_viewable(help_window)
        help_window.bind("<Destroy>", lambda e: e.widget is help_window
                         and _grab_when_viewable(parent))

        help_window.grid_rowconfigure(0, weight=1)
        help_window.grid_columnconfigure(0, weight=1)

        help_textbox = tk.Text(help_window, wrap=tk.NONE, font=("TkFixedFont",), **OUTLINE)
        help_textbox.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
        help_textbox.insert(tk.END, help_text)
        help_textbox.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=help_textbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=(10, 0))
        help_textbox.config(yscrollcommand=scrollbar.set)

        ttk.Label(
            help_window,
            text=f"The GUI always passes -m <model> -c {CTX_SIZE} before your options, "
                 "and --host/--port after them.",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 0))

        ttk.Button(help_window, text="Close", command=help_window.destroy).grid(
            row=2, column=0, columnspan=2, pady=10)

    def show_editor_window(self, _, inner_label):
        if self.editor_window and self.editor_window.winfo_exists():
            self.editor_window.lift()
            return
        if self.interface.busy("editing"):
            return

        editor_window = tk.Toplevel(self.interface.root)
        editor_window.title("Chat Editor")

        # Remove absolute positioning
        editor_window.geometry("400x300")
        
        # Assign it as a transient dialog
        editor_window.transient(self.interface.root)
        
        # Tell the Linux WM to treat this as a dialog/utility (skips taskbar)
        if platform.system().lower() == "linux":
            editor_window.attributes("-type", "dialog")
        
        _grab_when_viewable(editor_window)

        # Updated chat_editor
        chat_editor = tk.Text(
            editor_window,
            **OUTLINE,
        )
        chat_editor.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        chat_editor.insert(tk.END, inner_label.get_text())

        editor_window.grid_rowconfigure(0, weight=1)
        editor_window.grid_columnconfigure(0, weight=1)
        editor_window.grid_columnconfigure(1, weight=1)

        def _save():
            text = chat_editor.get("1.0", "end-1c")
            editor_window.destroy()
            self.interface.edit_message(inner_label, text)

        save_button = ttk.Button(editor_window, text="Save", command=_save)
        save_button.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        cancel_button = ttk.Button(
            editor_window, text="Cancel", command=editor_window.destroy
        )
        cancel_button.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        editor_window.grid_columnconfigure(0, weight=1, uniform="btn")
        editor_window.grid_columnconfigure(1, weight=1, uniform="btn")

        self.editor_window = editor_window

    def create_inner_label(self, on_right_side: bool = False):
        chat_box = self.interface.chat_box
        if self.bubble_font is None:
            self.bubble_font = font.Font(family=self.interface.default_font, size=12)
            self.markdown_styles = MarkdownStyles(self.bubble_font)
        width = chat_box.winfo_width()
        inner_label = ChatBubble(
            chat_box,
            self.bubble_font,
            max_width=int((width if width > 1 else chat_box.winfo_reqwidth()) * 0.7),
            background="#48a4f2" if on_right_side else "#eaeaea",
            foreground="white" if on_right_side else "black",
            select_background="#1f5f99" if on_right_side else "#9cc8f0",
            on_resize=self.interface.request_scroll,
            # Only model replies are markdown, the user's own text is shown as typed
            markdown=None if on_right_side else self.markdown_styles,
        )
        self.interface.label_widgets.append(inner_label)

        self.interface.bind_chat_wheel(inner_label.text)

        _right_menu = tk.Menu(
            inner_label, tearoff=0,
            postcommand=lambda: _right_menu.entryconfig(
                "Copy Selection", state="normal" if inner_label.selected_text() else "disabled"),
        )
        _right_menu.add_command(
            label="Edit", command=lambda: self.show_editor_window(None, inner_label)
        )
        _right_menu.add_command(
            label="Copy Selection",
            command=lambda: self.interface.copy_text(inner_label.selected_text())
        )
        _right_menu.add_command(
            label="Copy This", command=lambda: self.interface.copy_text(inner_label.get_text())
        )
        _right_menu.add_separator()
        _right_menu.add_command(label="Clear Chat", command=self.interface.clear_chat)
        _right_click = (
            "<Button-2>" if platform.system().lower() == "darwin" else "<Button-3>"
        )
        inner_label.text.bind(_right_click, lambda e: self.show_popup(_right_menu, e))
        chat_box.window_create(tk.END, window=inner_label)
        if on_right_side:
            idx = self.interface.chat_box.index("end-1c").split(".")[0]
            self.interface.chat_box.tag_add("Right", f"{idx}.0", f"{idx}.end")


def run():
    _prefer_color_emoji()
    root = tk.Tk()
    
    ttk.Style().theme_use('breeze')

    # Flatten the 3D borders on the menubar and right-click menus
    root.option_add("*Menu.relief", "flat")
    root.option_add("*Menu.borderWidth", 1)
    root.option_add("*Menu.activeBorderWidth", 0)

    root.title("Kappai GUI")
    root.tk.call('tk', 'appname', 'kappai-gui')
    root.geometry("800x600")

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=0)

    app = KappaiInterface(root)

    app.chat_box.tag_configure(
        "Bold", foreground="#ff007b", font=(app.default_font, 10, "bold")
    )
    app.chat_box.tag_configure("Error", foreground="red")
    app.chat_box.tag_configure("Right", justify="right")

    def _on_close():
        app.stop_server()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    run()
