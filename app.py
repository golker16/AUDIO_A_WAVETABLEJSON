#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox


# Reusamos exactamente el pipeline de export ya existente:
# wav_to_frames_and_meta() + build_wtgen_json_spectral()
# (es lo mismo que hace app3.py en cmd_export) :contentReference[oaicite:1]{index=1}
from app1 import (
    DEFAULT_TABLE_SIZE,
    DEFAULT_FRAMES,
    DEFAULT_SR,
    DEFAULT_HARMONICS,
    DEFAULT_NOISE_BANDS,
    wav_to_frames_and_meta,
    build_wtgen_json_spectral,
)


@dataclass
class ExportDefaults:
    sr: int = DEFAULT_SR
    frames: int = DEFAULT_FRAMES
    table_size: int = DEFAULT_TABLE_SIZE
    seed: int = 1
    engine_name: str = "wt_exe"
    engine_version: str = "1.2.0"
    preset_name: str = "FromWavPeak_SpectralData"
    harmonics: int = DEFAULT_HARMONICS
    noise_bands: int = DEFAULT_NOISE_BANDS


def export_wav_to_json(in_wav: Path, out_dir: Path, d: ExportDefaults) -> Path:
    in_wav = in_wav.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not in_wav.exists():
        raise FileNotFoundError(f"Input no existe: {in_wav}")

    # Nombre: mismo base name, pero .json (no .wtgen.json)
    out_path = out_dir / f"{in_wav.stem}.json"

    frames, meta = wav_to_frames_and_meta(
        wav_path=in_wav,
        sr_target=int(d.sr),
        frames_n=int(d.frames),
        table_size=int(d.table_size),
    )

    doc = build_wtgen_json_spectral(
        frames=frames,
        meta=meta,
        engine_name=d.engine_name,
        engine_version=d.engine_version,
        preset_name=d.preset_name,
        seed=int(d.seed),
        table_size=int(d.table_size),
        frames_n=int(d.frames),
        harm_count=int(d.harmonics),
        noise_bands=int(d.noise_bands),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    return out_path


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WTGEN Export -> JSON")
        self.geometry("720x420")
        self.minsize(640, 380)

        self.defaults = ExportDefaults()
        self.selected_files: List[Path] = []
        self.out_dir: Optional[Path] = None

        # Cola para logs desde thread
        self.q: "Queue[Tuple[str, str]]" = Queue()

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        root = tk.Frame(self, padx=12, pady=12)
        root.pack(fill="both", expand=True)

        # --- Output dir row
        row1 = tk.Frame(root)
        row1.pack(fill="x")

        tk.Label(row1, text="Carpeta de salida:").pack(side="left")
        self.out_var = tk.StringVar(value="")
        self.out_entry = tk.Entry(row1, textvariable=self.out_var)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=8)

        tk.Button(row1, text="Elegir...", command=self.choose_out_dir).pack(side="left")

        # --- Buttons row
        row2 = tk.Frame(root)
        row2.pack(fill="x", pady=(10, 0))

        tk.Button(row2, text="Agregar 1 archivo", command=self.add_one).pack(side="left")
        tk.Button(row2, text="Agregar varios (batch)", command=self.add_many).pack(side="left", padx=8)
        tk.Button(row2, text="Limpiar lista", command=self.clear_list).pack(side="left")

        self.run_btn = tk.Button(row2, text="Procesar", command=self.run_export)
        self.run_btn.pack(side="right")

        # --- Files list
        tk.Label(root, text="Archivos seleccionados:").pack(anchor="w", pady=(12, 4))
        self.listbox = tk.Listbox(root, height=10)
        self.listbox.pack(fill="both", expand=True)

        # --- Log
        tk.Label(root, text="Log:").pack(anchor="w", pady=(12, 4))
        self.log = tk.Text(root, height=6, wrap="word")
        self.log.pack(fill="both", expand=False)

    def choose_out_dir(self) -> None:
        d = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if d:
            self.out_dir = Path(d)
            self.out_var.set(str(self.out_dir))

    def _add_files(self, paths: List[str]) -> None:
        for p in paths:
            pp = Path(p)
            if pp.suffix.lower() != ".wav":
                # si quieres aceptar más tipos, cambia esto
                continue
            if pp not in self.selected_files:
                self.selected_files.append(pp)
                self.listbox.insert("end", str(pp))

        self._log(f"Archivos en cola: {len(self.selected_files)}")

    def add_one(self) -> None:
        p = filedialog.askopenfilename(
            title="Selecciona 1 WAV",
            filetypes=[("WAV", "*.wav"), ("Todos", "*.*")]
        )
        if p:
            self._add_files([p])

    def add_many(self) -> None:
        ps = filedialog.askopenfilenames(
            title="Selecciona varios WAV",
            filetypes=[("WAV", "*.wav"), ("Todos", "*.*")]
        )
        if ps:
            self._add_files(list(ps))

    def clear_list(self) -> None:
        self.selected_files.clear()
        self.listbox.delete(0, "end")
        self._log("Lista limpiada.")

    def run_export(self) -> None:
        out_txt = self.out_var.get().strip()
        if not out_txt:
            messagebox.showerror("Falta carpeta", "Selecciona una carpeta de salida.")
            return
        self.out_dir = Path(out_txt)

        if not self.selected_files:
            messagebox.showerror("Faltan archivos", "Agrega al menos 1 archivo WAV.")
            return

        self.run_btn.config(state="disabled")
        self._log("Iniciando export...")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        ok = 0
        fail = 0
        out_dir = self.out_dir or Path(".")

        for i, wav in enumerate(self.selected_files, start=1):
            try:
                self.q.put(("log", f"[{i}/{len(self.selected_files)}] Procesando: {wav.name}"))
                out_json = export_wav_to_json(wav, out_dir, self.defaults)
                self.q.put(("log", f"  -> OK: {out_json.name}"))
                ok += 1
            except Exception as e:
                self.q.put(("log", f"  -> ERROR: {wav.name} :: {e}"))
                fail += 1

        self.q.put(("done", f"Listo. OK={ok}  ERROR={fail}  (salida: {out_dir})"))

    def _poll_queue(self) -> None:
        try:
            while True:
                typ, msg = self.q.get_nowait()
                if typ == "log":
                    self._log(msg)
                elif typ == "done":
                    self._log(msg)
                    self.run_btn.config(state="normal")
        except Empty:
            pass
        self.after(120, self._poll_queue)

    def _log(self, s: str) -> None:
        self.log.insert("end", s + "\n")
        self.log.see("end")


def main_gui() -> None:
    app = App()
    app.mainloop()


def main() -> None:
    # Si lo ejecutas con args, conserva el CLI actual (app3.py) :contentReference[oaicite:2]{index=2}
    # Ejemplo: python app.py export input.wav -o out.json
    if len(sys.argv) > 1:
        from app3 import main as cli_main
        cli_main()
        return

    # Sin args: GUI
    main_gui()


if __name__ == "__main__":
    main()
