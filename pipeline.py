"""
OX RIG V2 — Engine Pipeline
============================
The main loop that produces heartbeats, fetches data, runs divergence
detection, and emits alerts.

This script is launched by api_node.py as a subprocess. Its stdout is
captured by the SidecarLogger and relayed to the parent console.

IMPORTANT: All print() calls use flush=True as a belt-and-suspenders
measure alongside PYTHONUNBUFFERED=1 set by the parent process.
"""

import sys
import time
import signal
from datetime import datetime
import pytz

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEZONE = pytz.timezone('America/Los_Angeles')
HEARTBEAT_INTERVAL = 5  # seconds between heartbeats
DATA_FETCH_INTERVAL = 60  # seconds between data refreshes

# =============================================================================
# UTILITIES
# =============================================================================

def now_pst():
    """Get current time in PST/PDT as a formatted string."""
    return datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')


def log(msg, level='INFO'):
    """
    Structured log line. Every line is flushed immediately so the
    SidecarLogger in api_node.py can relay it in real time.
    """
    print(f'[{now_pst()}] [{level}] {msg}', flush=True)


# =============================================================================
# ENGINE LOOP
# =============================================================================

class EnginePipeline:
    """
    The V2 engine pipeline. Runs continuously, emitting:
      - Heartbeats every HEARTBEAT_INTERVAL seconds
      - Data fetch logs every DATA_FETCH_INTERVAL seconds
      - Divergence alerts when detected
    """

    def __init__(self):
        self.running = True
        self.tick = 0
        self.last_fetch = 0

    def start(self):
        log('Engine pipeline starting...')
        log(f'Timezone locked to: {TIMEZONE}')
        log(f'Heartbeat interval: {HEARTBEAT_INTERVAL}s')
        log(f'Data fetch interval: {DATA_FETCH_INTERVAL}s')
        log('=' * 60)

        self._print_banner()

        while self.running:
            try:
                self._tick()
                time.sleep(HEARTBEAT_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log(f'Pipeline error: {e}', level='ERROR')
                time.sleep(1)

        log('Engine pipeline stopped.')

    def stop(self):
        self.running = False

    def _print_banner(self):
        banner = r"""
   ____  __  __   ____  ___ ____   __     _____
  / __ \ \ \/ /  / __ \|_ _/ ___| \ \   / /___ \
 | |  | | \  /  | |__) || | |  _   \ \ / /  __) |
 | |  | | /  \  |  _  / | | |_| |   \ V /  / __/
 |  __/ /_/\_\ |_| \_\|___\____|    \_/  |_____|
  \___/
        """
        for line in banner.strip().split('\n'):
            print(line, flush=True)
        print(flush=True)

    def _tick(self):
        self.tick += 1
        now = time.time()

        # Heartbeat
        log(f'HEARTBEAT #{self.tick} | uptime={self.tick * HEARTBEAT_INTERVAL}s')

        # Periodic data fetch
        if now - self.last_fetch >= DATA_FETCH_INTERVAL:
            self._fetch_data()
            self.last_fetch = now

    def _fetch_data(self):
        """Simulate a data fetch cycle."""
        log('Fetching latest candle data...', level='DATA')
        # In production, this would call an exchange API
        time.sleep(0.1)  # Simulate network latency
        log('Candle data refreshed (500 bars)', level='DATA')

        # Run divergence scan
        self._scan_divergences()

    def _scan_divergences(self):
        """Simulate divergence detection."""
        log('Running divergence scan...', level='SCAN')
        time.sleep(0.05)  # Simulate computation

        # In production, this calls detect_divergences() from divergence_detector.py
        log('Scan complete: 0 new divergences', level='SCAN')


# =============================================================================
# MAIN
# =============================================================================

pipeline = None

def shutdown(signum, frame):
    log(f'Received signal {signum}, shutting down...', level='SHUTDOWN')
    if pipeline:
        pipeline.stop()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pipeline = EnginePipeline()
    pipeline.start()
