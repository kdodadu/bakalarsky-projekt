"""Local capacity tooling for the pinned Windows llama.cpp b12602 C ABI.

Uses the installed tokenizer/vision preprocessing, never a Python approximation.
The language model is vocabulary-only; no encoder/decoder evaluation is called.
"""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


class ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int), ("load_mode", C.c_int),
                ("lazy_mode", C.c_int), ("main_gpu", C.c_int32), ("tensor_split", C.c_void_p),
                ("progress_callback", C.c_void_p), ("progress_callback_user_data", C.c_void_p),
                ("kv_overrides", C.c_void_p), ("vocab_only", C.c_bool), ("check_tensors", C.c_bool),
                ("use_extra_bufts", C.c_bool), ("no_host", C.c_bool), ("no_alloc", C.c_bool), ("load_mtp", C.c_bool)]


class VisionParams(C.Structure):
    _fields_ = [("use_gpu", C.c_bool), ("device", C.c_void_p), ("print_timings", C.c_bool),
                ("n_threads", C.c_int), ("image_marker", C.c_char_p), ("media_marker", C.c_char_p),
                ("flash_attn_type", C.c_int), ("warmup", C.c_bool),
                ("image_min_tokens", C.c_int), ("image_max_tokens", C.c_int),
                ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p), ("batch_max_tokens", C.c_int32),
                ("progress_callback", C.c_void_p), ("progress_callback_user_data", C.c_void_p)]


class VideoParams(C.Structure):
    _fields_ = [("fps_target", C.c_float), ("ffmpeg_bin_dir", C.c_char_p),
                ("timestamp_interval_ms", C.c_int64), ("no_soundtrack", C.c_bool), ("decoder", C.c_void_p)]


class InitOptions(C.Structure):
    _fields_ = [("video_params", VideoParams)]


class BitmapResult(C.Structure):
    _fields_ = [("bitmap", C.c_void_p), ("video_ctx", C.c_void_p)]


class InputText(C.Structure):
    _fields_ = [("text", C.c_char_p), ("text_len", C.c_size_t), ("add_special", C.c_bool), ("parse_special", C.c_bool)]


class NativeCounter:
    def __init__(self, model_path, projector_path, *, image_min_tokens=-1, image_max_tokens=-1):
        if os.name != "nt" or C.sizeof(ModelParams) != 80 or C.sizeof(VisionParams) != 96:
            raise RuntimeError("Capacity native bindings require the inspected b12602 Windows x64 ABI.")
        executable = shutil.which("llama")
        if not executable:
            raise RuntimeError("llama.exe is not on PATH")
        version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15)
        if "build 12602, commit 418fe97b5" not in version.stdout + version.stderr:
            raise RuntimeError("Native capacity ABI is pinned to llama.cpp b12602-418fe97b5")
        self.directory = Path(executable).parent
        self.dll_search = os.add_dll_directory(str(self.directory))
        self.ggml = C.CDLL(str(self.directory / "ggml.dll"))
        self.llama = C.CDLL(str(self.directory / "llama.dll"))
        self.mtmd = C.CDLL(str(self.directory / "mtmd.dll"))
        self.log_lines = []
        self.log_callback = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)(
            lambda level, text, user: self.log_lines.append(text.decode("utf-8", errors="replace")) if level >= 4 else None)
        self.llama.llama_log_set.argtypes = [C.c_void_p, C.c_void_p]
        self.llama.llama_log_set(self.log_callback, None)
        self.mtmd.mtmd_helper_log_set.argtypes = [C.c_void_p, C.c_void_p]
        self.mtmd.mtmd_helper_log_set(self.log_callback, None)
        def function(lib, name, result, *args):
            method = getattr(lib, name)
            method.restype, method.argtypes = result, list(args)
            return method
        function(self.ggml, "ggml_backend_load_all_from_path", None, C.c_char_p)(str(self.directory).encode())
        self.model_default = function(self.llama, "llama_model_default_params", ModelParams)
        self.model_load = function(self.llama, "llama_model_load_from_file", C.c_void_p, C.c_char_p, ModelParams)
        self.model_free = function(self.llama, "llama_model_free", None, C.c_void_p)
        self.vision_default = function(self.mtmd, "mtmd_context_params_default", VisionParams)
        self.vision_load = function(self.mtmd, "mtmd_init_from_file", C.c_void_p, C.c_char_p, C.c_void_p, VisionParams)
        self.vision_free = function(self.mtmd, "mtmd_free", None, C.c_void_p)
        self.options = function(self.mtmd, "mtmd_helper_init_opt_default", InitOptions)()
        self.bitmap_load = function(self.mtmd, "mtmd_helper_bitmap_init_from_buf", BitmapResult,
                                    C.c_void_p, C.c_char_p, C.c_size_t, C.c_bool, InitOptions)
        self.bitmap_make = function(self.mtmd, "mtmd_bitmap_init", C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p)
        self.bitmap_free = function(self.mtmd, "mtmd_bitmap_free", None, C.c_void_p)
        self.bitmap_width = function(self.mtmd, "mtmd_bitmap_get_nx", C.c_uint32, C.c_void_p)
        self.bitmap_height = function(self.mtmd, "mtmd_bitmap_get_ny", C.c_uint32, C.c_void_p)
        self.chunks_new = function(self.mtmd, "mtmd_input_chunks_init", C.c_void_p)
        self.chunks_free = function(self.mtmd, "mtmd_input_chunks_free", None, C.c_void_p)
        self.chunks_size = function(self.mtmd, "mtmd_input_chunks_size", C.c_size_t, C.c_void_p)
        self.chunk_get = function(self.mtmd, "mtmd_input_chunks_get", C.c_void_p, C.c_void_p, C.c_size_t)
        self.chunk_kind = function(self.mtmd, "mtmd_input_chunk_get_type", C.c_int, C.c_void_p)
        self.chunk_tokens = function(self.mtmd, "mtmd_input_chunk_get_n_tokens", C.c_size_t, C.c_void_p)
        self.n_tokens = function(self.mtmd, "mtmd_helper_get_n_tokens", C.c_size_t, C.c_void_p)
        self.n_positions = function(self.mtmd, "mtmd_helper_get_n_pos", C.c_int32, C.c_void_p)
        self.tokenize = function(self.mtmd, "mtmd_tokenize", C.c_int32, C.c_void_p, C.c_void_p,
                                 C.POINTER(InputText), C.POINTER(C.c_void_p), C.c_size_t)
        self.marker = function(self.mtmd, "mtmd_default_marker", C.c_char_p)().decode()
        params = self.model_default()
        if params.n_gpu_layers not in (-1, 0, 99) or params.load_mode not in range(-1, 5):
            raise RuntimeError("Unexpected model params ABI; refusing native calls")
        params.vocab_only, params.n_gpu_layers = True, 0
        self.model = self.model_load(str(model_path).encode(), params)
        if not self.model:
            raise RuntimeError("Native vocabulary-only load failed")
        vision = self.vision_default()
        vision.use_gpu, vision.warmup, vision.print_timings, vision.n_threads = False, False, False, 4
        vision.image_min_tokens, vision.image_max_tokens = image_min_tokens, image_max_tokens
        self.context = self.vision_load(str(projector_path).encode(), self.model, vision)
        if not self.context:
            self.model_free(self.model)
            raise RuntimeError("Native vision metadata/preprocessor initialization failed")

    def decode(self, path):
        raw = Path(path).read_bytes()
        result = self.bitmap_load(self.context, raw, len(raw), True, self.options)
        if not result.bitmap:
            return {"decodable": False, "width": None, "height": None}
        try:
            return {"decodable": True, "width": self.bitmap_width(result.bitmap), "height": self.bitmap_height(result.bitmap)}
        finally:
            self.bitmap_free(result.bitmap)

    def count(self, rendered_prompt, width, height):
        raw = rendered_prompt.encode("utf-8")
        text = InputText(raw, len(raw), True, True)
        bitmap = self.bitmap_make(width, height, None)
        bitmaps = (C.c_void_p * 1)(bitmap)
        chunks = self.chunks_new()
        try:
            result = self.tokenize(self.context, chunks, C.byref(text), bitmaps, 1)
            if result:
                raise RuntimeError(f"Native multimodal tokenization failed: {result}")
            kinds = {}
            for index in range(self.chunks_size(chunks)):
                chunk = self.chunk_get(chunks, index)
                kind = str(self.chunk_kind(chunk))
                kinds[kind] = kinds.get(kind, 0) + self.chunk_tokens(chunk)
            return {"input_tokens": self.n_tokens(chunks), "input_positions": self.n_positions(chunks), "chunk_tokens": kinds}
        finally:
            self.chunks_free(chunks)
            self.bitmap_free(bitmap)

    def close(self):
        if self.context:
            self.vision_free(self.context)
            self.context = None
        if self.model:
            self.model_free(self.model)
            self.model = None


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_capacity_index(config, manifest, root):
    """Opt-in exact cached bounds; never drop a record or alter an input."""
    check = config["capacity_preflight"]
    path = (root / check["path"]).resolve()
    if digest_file(path) != check["sha256"]:
        raise ValueError("Capacity cache changed; rerun the separate technical audit.")
    index = json.loads(path.read_text(encoding="utf-8"))
    proof = index["proof"]
    experiment = config["experiment"]
    expected = {"gguf_sha256":experiment["gguf"]["sha256"], "prompt_sha256":config["protocol"]["prompt_sha256"],
                "news_manifest_sha256":digest_file(root / config["prepared_dir"] / "manifest.json")}
    if (index["model"] != config["model_id"] or index["historical_mismatches"]
            or any(proof.get(k)!=v for k,v in expected.items())
            or proof["mmproj_sha256"] not in {v["sha256"] for v in experiment["mmproj"]}):
        raise ValueError("Capacity cache belongs to another model, prompt, images or input manifest.")
    image_settings = experiment["image_processing"]
    if any(proof.get(k) != image_settings[k] for k in ("image_min_tokens","image_max_tokens","client_resize")):
        raise ValueError("Capacity cache uses another image preprocessing protocol.")
    executable = shutil.which("llama")
    if not executable:
        raise ValueError("Capacity preflight requires the calibrated llama.exe and DLLs on PATH.")
    directory = Path(executable).parent
    if any(digest_file(directory / name) != digest for name,digest in proof["dll_sha256"].items()):
        raise ValueError("Backend DLLs differ from the calibrated capacity counter.")
    records = [r for r in index["records"] if r["dataset"]=="NewsCLIPpings"]
    mapping = {int(r["input_id"].split(":")[1]):r for r in records}
    if len(mapping)!=len(records) or set(mapping)!={r["annotation_index"] for r in manifest["records"]}:
        raise ValueError("Capacity cache must cover every unique annotation_index, without duplicates.")
    counted, unknown, maximum = 0, 0, 0
    for row in manifest["records"]:
        saved = mapping[row["annotation_index"]]
        if (saved["image_sha256"]!=row["image_sha256"]
                or saved["caption_sha256"]!=hashlib.sha256(row["caption"].encode()).hexdigest()):
            raise ValueError(f"Capacity input changed: annotation_index={row['annotation_index']}")
        if "input_tokens" not in saved:
            # Native decoder failures stay in the normal queue and retain the
            # existing ERROR policy. A missing length is never a guessed zero.
            if saved["decodable"]:
                raise ValueError(f"Missing native count: annotation_index={row['annotation_index']}")
            unknown += 1
            continue
        length = max(saved["input_tokens"],saved["input_positions"])
        required = length + config["max_tokens"] + check["margin_tokens"]
        if required > experiment["context_size"]:
            raise ValueError(f"Capacity preflight: annotation_index={row['annotation_index']}, native input={length} + output={config['max_tokens']} + margin={check['margin_tokens']} = {required} exceeds context={experiment['context_size']}. No input was truncated or skipped.")
        counted += 1
        maximum = max(maximum,length)
    return {"counted_records":counted,"native_decoder_failures_without_length":unknown,"max_input_tokens":maximum,
            "output_tokens":config["max_tokens"],"margin_tokens":check["margin_tokens"],
            "policy":"All records stay queued; decoder failures retain the existing ERROR accounting."}
