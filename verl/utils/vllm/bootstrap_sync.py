# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Startup-only coordination, independent of the vLLM engine's ready RPC."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BootstrapCoordinator:
    """Gate profiling until all first-weight receivers and senders have finished.

    Tensor data still travels over the existing bucketed IPC/SHM channel. HTTP
    only carries the small ready/loaded/release messages. Each waiting receiver
    gets its own request thread, so one rank cannot block another rank's ACK.
    """

    def __init__(self, worker_ids, *, host="127.0.0.1", advertise_host=None, timeout_seconds=300):
        worker_ids = list(worker_ids)
        if not worker_ids or len(set(worker_ids)) != len(worker_ids):
            raise ValueError("Bootstrap worker_ids must be nonempty and unique")
        if any(not isinstance(worker_id, str) or not worker_id for worker_id in worker_ids):
            raise ValueError("Bootstrap worker_ids must be nonempty strings")
        if timeout_seconds <= 0:
            raise ValueError("Bootstrap timeout_seconds must be positive")
        self.worker_ids = frozenset(worker_ids)
        self.timeout_seconds = float(timeout_seconds)
        self._condition = threading.Condition()
        self._ready = set()
        self._loaded = set()
        self._released = False
        self._error = None
        self._closed = False
        coordinator = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    coordinator._receive(self.path, body)
                    status, response = 200, {"ok": True}
                except (RuntimeError, TimeoutError, ValueError, KeyError) as error:
                    status, response = 400, {"error": str(error)}
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The driver may already have terminated an aborted worker.

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._server.daemon_threads = True
        advertised = advertise_host or ("127.0.0.1" if host == "0.0.0.0" else host)
        self.address = f"http://{advertised}:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="verl-bootstrap")
        self._thread.start()

    def serializable_config(self):
        return {"address": self.address, "timeout_seconds": self.timeout_seconds}

    def _raise_if_aborted(self):
        if self._error is not None:
            raise RuntimeError(f"Initial weight synchronization aborted: {self._error}")

    def _wait(self, predicate, description):
        deadline = time.monotonic() + self.timeout_seconds
        with self._condition:
            while not predicate():
                self._raise_if_aborted()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._error = f"Timed out waiting for {description}"
                    self._condition.notify_all()
                    raise TimeoutError(self._error)
                self._condition.wait(remaining)
            self._raise_if_aborted()

    def wait_ready(self):
        self._wait(lambda: self._ready == self.worker_ids, "bootstrap receivers to be ready")

    def wait_loaded(self):
        self._wait(lambda: self._loaded == self.worker_ids, "bootstrap receivers to load all weights")

    def release(self):
        """The caller must also await sender cleanup/offload before this call."""
        with self._condition:
            self._raise_if_aborted()
            if self._loaded != self.worker_ids:
                raise RuntimeError("Cannot release bootstrap before every receiver has loaded weights")
            self._released = True
            self._condition.notify_all()

    def abort(self, error):
        with self._condition:
            if self._error is None:
                self._error = str(error)
            self._condition.notify_all()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self._released:
            self.abort("Bootstrap coordinator closed")
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _receive(self, path, body):
        worker_id = body["worker_id"]
        if worker_id not in self.worker_ids:
            raise ValueError(f"Unknown bootstrap worker: {worker_id}")
        with self._condition:
            self._raise_if_aborted()
            if path == "/ready":
                self._ready.add(worker_id)
            elif path == "/loaded":
                if worker_id not in self._ready:
                    raise RuntimeError(f"Bootstrap worker {worker_id} reported loaded before ready")
                self._loaded.add(worker_id)
            elif path == "/failed":
                self.abort(f"{worker_id}: {body.get('error', 'Receiver failed')}")
            elif path != "/wait-release":
                raise ValueError(f"Unknown bootstrap operation: {path}")
            self._condition.notify_all()
        if path == "/wait-release":
            self._wait(lambda: self._released, "driver to release bootstrap receivers")


def _bootstrap_request(config, operation, worker_id, **kwargs):
    payload = json.dumps({"worker_id": worker_id, **kwargs}).encode()
    request = urllib.request.Request(
        config["address"].rstrip("/") + "/" + operation,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Do not send localhost coordination requests through HTTP proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=float(config.get("timeout_seconds", 300)) + 5) as response:
            json.load(response)
    except urllib.error.HTTPError as error:
        message = json.load(error).get("error", str(error))
        raise RuntimeError(message) from error


def bootstrap_receiver_ready(config, worker_id):
    _bootstrap_request(config, "ready", worker_id)


def bootstrap_receiver_loaded(config, worker_id):
    _bootstrap_request(config, "loaded", worker_id)


def bootstrap_wait_for_release(config, worker_id):
    _bootstrap_request(config, "wait-release", worker_id)


def bootstrap_receiver_failed(config, worker_id, error):
    _bootstrap_request(config, "failed", worker_id, error=str(error))
