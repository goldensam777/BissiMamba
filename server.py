#!/usr/bin/env python3
"""
BissiMamba Web Server — Python stdlib only, no external dependencies.

Usage:
    python3 server.py [port]          (default port: 8080)

Endpoints:
    GET  /              -> static/index.html
    GET  /static/*      -> static files
    POST /api/chat      -> JSON {message: str} -> {reply: str, error?: str}
    GET  /api/status    -> {trained: bool, model_path: str}
"""

import http.server
import json
import os
import subprocess
import sys

PORT        = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
CHAT_BIN    = "./mamba_chat"
MODEL_PATH  = "lm_checkpoint.bin"
STATIC_DIR  = "static"
TIMEOUT_SEC = 60   # generation timeout

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css",
    ".js":   "application/javascript",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
}


class BissiHandler(http.server.BaseHTTPRequestHandler):

    # ------------------------------------------------------------------ #
    # GET
    # ------------------------------------------------------------------ #
    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_file(os.path.join(STATIC_DIR, "index.html"))
        elif path == "/api/status":
            self._send_json({
                "trained":    os.path.isfile(MODEL_PATH),
                "model_path": MODEL_PATH,
                "chat_bin":   CHAT_BIN,
                "chat_ready": os.path.isfile(CHAT_BIN),
            })
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            self._serve_file(os.path.join(STATIC_DIR, rel))
        else:
            self._send_error(404, "Not found")

    # ------------------------------------------------------------------ #
    # POST
    # ------------------------------------------------------------------ #
    def do_POST(self):
        if self.path != "/api/chat":
            self._send_error(404, "Not found")
            return

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data    = json.loads(body)
            message = str(data.get("message", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if not message:
            self._send_json({"error": "Empty message"}, status=400)
            return

        # Check prerequisites
        if not os.path.isfile(CHAT_BIN):
            self._send_json({
                "error": (
                    "Chat binary not found. Build it first: make mamba_chat"
                )
            }, status=503)
            return

        if not os.path.isfile(MODEL_PATH):
            self._send_json({
                "error": (
                    "Model not trained yet. "
                    "Run: make mamba_lm_train && ./mamba_lm_train"
                )
            }, status=503)
            return

        # Spawn chat binary
        try:
            result = subprocess.run(
                [CHAT_BIN, MODEL_PATH],
                input=message,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
            )
            reply = result.stdout.strip()
            if not reply and result.stderr:
                reply = "[no output — check model training]"
            self._send_json({"reply": reply})
        except subprocess.TimeoutExpired:
            self._send_json({"error": "Generation timed out"}, status=504)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _serve_file(self, filepath):
        if not os.path.isfile(filepath):
            self._send_error(404, f"File not found: {filepath}")
            return
        ext  = os.path.splitext(filepath)[1]
        mime = MIME.get(ext, "application/octet-stream")
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status, msg):
        self._send_json({"error": msg}, status=status)

    def log_message(self, fmt, *args):
        # Simple clean log
        print(f"[{self.address_string()}] {fmt % args}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), BissiHandler)
    print(f"BissiMamba server running at http://localhost:{PORT}")
    print(f"  Chat binary : {CHAT_BIN}")
    print(f"  Model file  : {MODEL_PATH}")
    print(f"  Trained     : {os.path.isfile(MODEL_PATH)}")
    print(f"  Chat ready  : {os.path.isfile(CHAT_BIN)}")
    print()
    print("Workflow:")
    print("  1. make mamba_lm_train && ./mamba_lm_train   # train (200 epochs)")
    print("  2. make mamba_chat                            # build chat binary")
    print("  3. python3 server.py                          # start web server")
    print("  4. Open http://localhost:8080")
    print()
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
