# Kappai GUI

A simple desktop chat app for [kappai](https://github.com/System64fumo/kappai), a fast local AI inference engine.
One Python file using only Tkinter, with no other dependencies.

This is Botspot's fork of [chyok/ollama-gui](https://github.com/chyok/ollama-gui), switched from Ollama to kappai.
Install it from Pi-Apps:
[![badge](https://github.com/Botspot/pi-apps/blob/master/icons/badge.png?raw=true)](https://github.com/Botspot/pi-apps)

## Features

- Download models from Hugging Face, starting from a short list of recommended models. Models kappai can't run, or that are too large for your RAM, are hidden.
- Pick a model and the app starts `kappai-server` for you.
- Streaming replies with basic Markdown: bold, italic, code, headings, lists, quotes and links.
- Chats are saved automatically and the last one reopens on launch. Temporary chats are never saved.
- Edit any message. Editing your own message replaces everything below it with a new reply.
- Select and copy text from any message.

## Requirements

- Python 3 with Tkinter. On Raspberry Pi OS, Debian or Ubuntu: `sudo apt install python3-tk`
- kappai, [built from source](https://github.com/System64fumo/kappai#building). The app looks for `kappai-server` in this order:
  1. the `KAPPAI_SERVER` environment variable, e.g. `KAPPAI_SERVER=/opt/kappai/kappai-server`
  2. your `PATH`
  3. `~/kappai/build/kappai-server`

## Run

```
python3 kappai_gui.py
```

Open **Settings** to download your first model.

## Server options

Settings → **Server options** is passed to `kappai-server`. **Help** next to it lists every option.
The default is `--device cpu0`, because kappai's OpenGL backend currently crashes in the Raspberry Pi's graphics driver.
The app always passes `-c 8192` for the context size; add your own `-c` to override it.

## Where things are stored

| What | Where |
|---|---|
| Models | kappai's models folder (`models_dir` in `~/.config/sys64/kappai/tui.conf`), otherwise `~/models` |
| Chats | `~/.local/share/kappai-gui/chats/`, one JSON file per chat |
| Settings | `~/.config/kappai-gui/settings.json` |
| Model list cache | `~/.cache/kappai-gui/` |

Models are stored the same way kappai's own downloader stores them, so both tools share one library.

## Keeping up with kappai

kappai is changing quickly, and some of what it can do is built into this app rather than asked from kappai.
After a kappai update, these may be out of date until the app is updated too:

| What | If it's out of date | What you can do |
|---|---|---|
| Which model types kappai can run | Models kappai has just added support for don't appear in the list or in searches | Search for the model, then type its `owner/repo` into the search box to open it anyway |
| Which file sizes (quantizations) kappai can load | Newly supported files are hidden, and a file kappai no longer loads may still be listed | Pick a different file of the same model |
| The default server option `--device cpu0` | Models keep running on the CPU after kappai's graphics backend is fixed, or on a PC with a good GPU | Clear or change it under Settings → Server options |
| Where kappai keeps its models folder setting | The app falls back to `~/models` and no longer shares a model library with the kappai menu | Type your models folder into the **Models** box at the top of the window |
| The options used to start `kappai-server` | Choosing a model fails with "kappai-server exited", and the reason is printed in the terminal | Run the app from a terminal to see the error, and check Settings → Server options → **Help** |
| The Recommended list and model families | Newer models aren't in the Recommended list | Untick **Recommended**, or search for the model by name |

## Troubleshooting

**No module named tkinter**: install Tk for Python:
`sudo apt install python3-tk` (Debian/Ubuntu), `sudo dnf install python3-tkinter` (Fedora) or `brew install python-tk` (macOS).

**kappai-server not found**: see [Requirements](#requirements).

**macOS Sonoma: the window ignores clicks at startup**: this is a [Tcl/Tk bug](https://github.com/python/cpython/issues/110218), fixed in Python 3.11.7 and later.

## License

[MIT](LICENSE)
