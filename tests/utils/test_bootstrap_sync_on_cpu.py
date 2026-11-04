# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from verl.utils.vllm.bootstrap_sync import (
    BootstrapCoordinator,
    bootstrap_receiver_failed,
    bootstrap_receiver_loaded,
    bootstrap_receiver_ready,
    bootstrap_wait_for_release,
)


def test_all_receivers_ack_before_driver_releases_profiling():
    coordinator = BootstrapCoordinator(["r0", "r1"], timeout_seconds=3)
    config = coordinator.serializable_config()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            bootstrap_receiver_ready(config, "r0")
            # A waiting HTTP request must not block another rank's messages.
            waiting = executor.submit(bootstrap_wait_for_release, config, "r0")
            bootstrap_receiver_ready(config, "r1")
            coordinator.wait_ready()
            with pytest.raises(RuntimeError, match="before every receiver"):
                coordinator.release()
            bootstrap_receiver_loaded(config, "r0")
            bootstrap_receiver_loaded(config, "r1")
            coordinator.wait_loaded()
            assert not waiting.done()
            second = executor.submit(bootstrap_wait_for_release, config, "r1")
            coordinator.release()
            waiting.result(timeout=3)
            second.result(timeout=3)
    finally:
        coordinator.close()


def test_receiver_failure_wakes_driver_and_other_receivers():
    coordinator = BootstrapCoordinator(["r0", "r1"], timeout_seconds=3)
    config = coordinator.serializable_config()
    try:
        bootstrap_receiver_ready(config, "r0")
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(bootstrap_wait_for_release, config, "r0")
            bootstrap_receiver_failed(config, "r1", "bad weights")
            with pytest.raises(RuntimeError, match="bad weights"):
                coordinator.wait_loaded()
            with pytest.raises(RuntimeError, match="bad weights"):
                waiting.result(timeout=3)
    finally:
        coordinator.close()


def test_missing_receiver_times_out_and_release_stays_closed():
    coordinator = BootstrapCoordinator(["missing"], timeout_seconds=0.1)
    try:
        with pytest.raises(TimeoutError, match="receivers to be ready"):
            coordinator.wait_ready()
        with pytest.raises(RuntimeError, match="aborted"):
            coordinator.release()
    finally:
        coordinator.close()


def test_unknown_receiver_does_not_satisfy_ready():
    coordinator = BootstrapCoordinator(["r0"], timeout_seconds=3)
    try:
        with pytest.raises(RuntimeError, match="Unknown bootstrap worker"):
            bootstrap_receiver_ready(coordinator.serializable_config(), "other")
    finally:
        coordinator.close()


def test_control_and_plugin_modules_import_without_initializing_vllm():
    root = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import verl.utils.vllm.bootstrap_sync; "
            "import verl.utils.vllm.prequantized_loader; assert 'vllm' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": root},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
