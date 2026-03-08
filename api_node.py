"""
OX RIG V2 — API Node
====================
Serves the Mission Control dashboard and proxies data from the engine pipeline.

ROOT CAUSE FIX for Terminal Silence:

The V1 rig launched pipeline.py as a subprocess and inherited stdout directly,
so logs appeared in the console. The V2 refactor used subprocess.Popen with
default settings on Windows, which causes two problems:

  1. sys.stdout handle inheritance is blocked by Windows handle locks when
     the parent process is also writing to stdout.
  2. Python's print() buffers output by default in non-interactive (piped)
     mode, so even when the pipe is open, logs don't appear until the
     buffer fills up (typically 8KB).

THE FIX:
  - Launch the engine with subprocess.PIPE for both stdout and stderr.
  - Pass PYTHONUNBUFFERED=1 in the environment so the child process
    flushes every print() immediately.
  - Spin up a daemon threading.Thread that reads from the pipe line-by-line
    and writes to our own sys.stdout with flush=True.
  - This "sidecar logger" pattern works on both Windows and Linux and
    survives handle lock issues because the parent never inherits the
    child's stdout handle directly.
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

HOST = '0.0.0.0'
PORT = 8080
ENGINE_SCRIPT = 'pipeline.py'
STATIC_DIR = Path(__file__).parent  # Serve files from the repo root

# =============================================================================
# SIDECAR LOGGER — Fixes the Terminal Silence bug
# =============================================================================

class SidecarLogger(threading.Thread):
    """
    Reads lines from a subprocess pipe and relays them to the parent's stdout.

    This runs as a daemon thread so it dies automatically when the main
    process exits. Each line is flushed immediately so the user sees
    engine heartbeats and alerts in real time — no buffering.
    """

    def __init__(self, pipe, prefix='[ENGINE]', stream=None):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.prefix = prefix
        self.stream = stream or sys.stdout

    def run(self):
        try:
            for raw_line in iter(self.pipe.readline, b''):
                try:
                    line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
                except Exception:
                    line = str(raw_line)
                self.stream.write(f'{self.prefix} {line}\n')
                self.stream.flush()
        except (OSError, ValueError):
            # Pipe closed — engine exited
            pass
        finally:
            try:
                self.pipe.close()
            except Exception:
                pass


def launch_engine():
    """
    Start pipeline.py as a subprocess with proper log relay.

    Key details:
      - PYTHONUNBUFFERED=1 forces the child to flush prints immediately.
      - stdout and stderr are captured via PIPE, not inherited.
      - Two SidecarLogger threads relay output to the parent console.
    """
    engine_path = STATIC_DIR / ENGINE_SCRIPT
    if not engine_path.exists():
        print(f'[API NODE] WARNING: {ENGINE_SCRIPT} not found at {engine_path}', flush=True)
        return None

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    print(f'[API NODE] Launching engine: {engine_path}', flush=True)

    proc = subprocess.Popen(
        [sys.executable, str(engine_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(STATIC_DIR),
    )

    # Spin up sidecar loggers for stdout and stderr
    SidecarLogger(proc.stdout, prefix='[ENGINE]').start()
    SidecarLogger(proc.stderr, prefix='[ENGINE ERR]', stream=sys.stderr).start()

    print(f'[API NODE] Engine PID: {proc.pid}', flush=True)
    return proc


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class OxRigHandler(SimpleHTTPRequestHandler):
    """
    Serves static files (HTML, JS, CSS) and API endpoints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        """Override to add our prefix and flush."""
        print(f'[HTTP] {args[0]}', flush=True)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/health':
            self._json_response({'status': 'ok', 'engine_pid': engine_proc.pid if engine_proc else None})
            return

        if parsed.path == '/api/candles':
            self._serve_candles(parsed)
            return

        # Default: serve static files (unified_chart.html, etc.)
        super().do_GET()

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _serve_candles(self, parsed):
        """
        Proxy or generate candle data.
        In production this would forward to the engine's data store.
        For now, return demo data so the chart renders immediately.
        """
        params = parse_qs(parsed.query)
        limit = int(params.get('limit', ['500'])[0])

        now = int(time.time())
        interval = 60  # 1-minute candles
        price = 97500.0
        candles = []

        import random
        random.seed(42)  # Reproducible demo data

        for i in range(limit, 0, -1):
            t = now - (i * interval)
            change = (random.random() - 0.48) * 50
            o = price
            price += change
            c = price
            h = max(o, c) + random.random() * 30
            lo = min(o, c) - random.random() * 30
            v = 50 + random.random() * 200

            candles.append({
                'time': t,
                'open': round(o, 2),
                'high': round(h, 2),
                'low': round(lo, 2),
                'close': round(c, 2),
                'volume': round(v, 2),
            })

        self._json_response(candles)


# =============================================================================
# MAIN
# =============================================================================

engine_proc = None

def shutdown(signum, frame):
    print(f'\n[API NODE] Shutting down (signal {signum})...', flush=True)
    if engine_proc and engine_proc.poll() is None:
        engine_proc.terminate()
        try:
            engine_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            engine_proc.kill()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print('=' * 72, flush=True)
    print('  OX RIG V2 — API NODE', flush=True)
    print('=' * 72, flush=True)
    print(f'  Static dir : {STATIC_DIR}', flush=True)
    print(f'  Listen     : http://{HOST}:{PORT}', flush=True)
    print(f'  Dashboard  : http://localhost:{PORT}/unified_chart.html', flush=True)
    print('=' * 72, flush=True)

    # Launch the engine pipeline in the background
    engine_proc = launch_engine()

    # Start the HTTP server
    server = HTTPServer((HOST, PORT), OxRigHandler)
    print(f'[API NODE] Server listening on port {PORT}', flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(signal.SIGINT, None)
