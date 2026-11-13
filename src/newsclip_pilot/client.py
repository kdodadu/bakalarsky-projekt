"""HTTP client restricted to loopback, without proxies or redirects."""

import base64
from http.client import HTTPException
import ipaddress
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirects are disabled", headers, fp)


def loopback_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an http:// loopback server URL without credentials, query or fragment.")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        raise ValueError("Only a literal loopback address or localhost is allowed.") from None
    if not address.is_loopback or parsed.path.rstrip("/") not in ("", "/v1"):
        raise ValueError("Only a local loopback server, optionally ending in /v1, is allowed.")
    authority = f"[{host}]" if address.version == 6 else host
    if parsed.port is not None:
        authority += f":{parsed.port}"
    return urlunsplit(("http", authority, "", "", ""))


def image_data_url(path):
    path = Path(path).resolve()
    if "verite" in (part.lower() for part in path.parts):
        raise ValueError("VERITE must not be used in smoke tests or this pilot.")
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif raw.startswith(b"BM"):
        mime = "image/bmp"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image signature: {path.name}")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def make_payload(config, image, text, max_tokens=None):
    # Labels, identifiers, source, similarity and filenames never enter messages.
    return {
        "model": config["model"], "temperature": config["temperature"],
        "seed": config["seed"], "max_tokens": max_tokens or config["max_tokens"],
        "repeat_penalty": config["repeat_penalty"],
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "chat_template_kwargs": config["chat_template_kwargs"],
        "stream": False, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            {"type": "text", "text": text},
        ]}],
    }


class LocalClient:
    def __init__(self, base_url, timeout=180):
        self.base_url = loopback_url(base_url)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, endpoint, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(self.base_url + endpoint, data=data, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        result = {"response": None, "raw_http_response": None, "http_status": None, "error": None}
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result["http_status"] = response.status
                result["raw_http_response"] = response.read().decode("utf-8")
                result["response"] = json.loads(result["raw_http_response"])
        except HTTPError as exc:
            result["http_status"] = exc.code
            result["raw_http_response"] = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {exc.reason}"
            exc.close()
        except (URLError, OSError, ValueError, HTTPException) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

    def probe(self, expected_model):
        health = self.request("/health")
        models = self.request("/v1/models")
        for name, result in (("health", health), ("models", models)):
            if result["error"]:
                raise RuntimeError(f"Local server {name}: {result['error']}")
        if health["response"].get("status") != "ok":
            raise RuntimeError(f"Local server is not ready: {health['response']}")
        identifiers = {r.get("id") for r in models["response"].get("data", [])}
        if expected_model not in identifiers:
            raise RuntimeError(f"Expected {expected_model!r}; server reports {sorted(str(i) for i in identifiers)}")
        props = self.request("/props")
        return {"base_url": self.base_url, "health": health, "models": models, "props": props}


def interpret(result):
    parsed = {"raw_response": None, "prediction": None, "status": "request_error", "finish_reason": None}
    if result["error"]:
        return parsed
    try:
        choice = result["response"]["choices"][0]
        content = choice["message"]["content"]
        parsed.update(raw_response=content, finish_reason=choice.get("finish_reason"))
    except (TypeError, KeyError, IndexError):
        return parsed
    # Only outer whitespace is ignored. No substring, punctuation or case repair.
    if isinstance(content, str) and content.strip() in ("MATCH", "MISMATCH") and parsed["finish_reason"] == "stop":
        parsed.update(prediction=content.strip(), status="ok")
    else:
        parsed["status"] = "invalid_response"
    return parsed
