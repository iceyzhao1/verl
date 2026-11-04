# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU integration tests for the real bootstrap control service and driver ordering."""

import asyncio
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.vllm.bootstrap_sync import (
    BootstrapCoordinator,
    bootstrap_receiver_loaded,
    bootstrap_receiver_ready,
    bootstrap_wait_for_release,
)
from verl.utils.vllm.startup import (
    build_bootstrap_payload,
    create_prequantized_llm_server_manager,
    initialize_with_bootstrap,
)
from verl.workers.config.rollout import RolloutConfig


def test_startup_waits_for_all_ranks_and_sender_cleanup():
    async def run():
        events = []
        coordinator = BootstrapCoordinator(["rank-0", "rank-1"], timeout_seconds=3)
        config = coordinator.serializable_config()
        received = [asyncio.Event(), asyncio.Event()]

        async def receiver(rank):
            await asyncio.to_thread(bootstrap_receiver_ready, config, f"rank-{rank}")
            await received[rank].wait()
            await asyncio.to_thread(bootstrap_receiver_loaded, config, f"rank-{rank}")
            await asyncio.to_thread(bootstrap_wait_for_release, config, f"rank-{rank}")
            events.append(f"profile-{rank}")

        class Manager:
            async def _initialize_llm_servers(self):
                # Device queries must complete before the sender occupies the actors.
                events.append("device-queries")
                await asyncio.gather(receiver(0), receiver(1))

            async def _init_global_load_balancer(self):
                events.append("load-balancer")

        async def send(rank):
            events.append(f"send-{rank}")
            received[rank].set()
            # FSDP-style collectives require every rank to enter the send operation.
            await received[1 - rank].wait()
            await asyncio.sleep(0)
            events.append(f"cleanup-{rank}")

        def abort():
            pytest.fail("Successful startup must not kill actors")

        manager = Manager()
        result = await initialize_with_bootstrap(manager, coordinator, lambda: [send(0), send(1)], abort)
        assert result is manager
        assert events[0] == "device-queries"
        assert events[-1] == "load-balancer"
        for rank in (0, 1):
            assert events.index(f"profile-{rank}") > max(events.index("cleanup-0"), events.index("cleanup-1"))

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["initialize", "send"])
def test_failure_aborts_startup_and_closes_control_waiters(failure):
    async def run():
        coordinator = BootstrapCoordinator(["rank-0"], timeout_seconds=2)
        config = coordinator.serializable_config()
        aborted = []
        initialized = []

        class Manager:
            async def _initialize_llm_servers(self):
                if failure == "initialize":
                    raise RuntimeError("initialization failed")
                await asyncio.to_thread(bootstrap_receiver_ready, config, "rank-0")
                await asyncio.to_thread(bootstrap_wait_for_release, config, "rank-0")
                initialized.append(True)

            async def _init_global_load_balancer(self):
                pytest.fail("Failed startup must not publish a ready service")

        async def send():
            raise RuntimeError("send failed")

        with pytest.raises(RuntimeError, match="failed"):
            await initialize_with_bootstrap(Manager(), coordinator, lambda: [send()], lambda: aborted.append(True))
        assert aborted == [True]
        assert not initialized
        assert not coordinator._thread.is_alive()

    asyncio.run(run())


def test_payload_contains_restored_step_and_rejects_disagreeing_workers():
    info = {
        "worker_id": "replica-0-rank-0",
        "zmq_handle": "ipc:///tmp/test-rank-0",
        "use_shm": True,
        "quant_description": {"quant_method": "ascend", "layer.weight": "W8A8_MXFP8"},
        "manifest": [{"name": "layer.weight", "shape": [64, 32], "dtype": "float8_e4m3fn"}],
    }
    second = {**deepcopy(info), "worker_id": "replica-1-rank-0", "zmq_handle": "ipc:///tmp/test-rank-1"}
    result = build_bootstrap_payload([info, second], {"address": "http://127.0.0.1:1234"}, 100)
    assert result["global_steps"] == 100
    assert len(result["data_endpoints"]) == 2
    second["manifest"][0]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="manifest"):
        build_bootstrap_payload([info, second], {}, 100)
    with pytest.raises(ValueError, match="Duplicate"):
        build_bootstrap_payload([info, info], {}, 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "sglang"},
        {"qat_fake_quant": True},
        {"quantization": "fp8"},
        {"quantization_config_file": "/tmp/offline.json"},
        {"checkpoint_engine": {"backend": "nccl"}},
        {"layered_summon": True},
    ],
)
def test_new_mode_rejects_incompatible_configuration(overrides):
    with pytest.raises(ValueError, match="quantized_weight_sync"):
        RolloutConfig(**{"name": "vllm", "quantized_weight_sync": True, **overrides})


def test_old_mode_remains_default_and_new_mode_allows_null_mtp():
    assert RolloutConfig(name="vllm").quantized_weight_sync is False
    assert RolloutConfig(name="vllm", quantized_weight_sync=True, mtp=None).quantized_weight_sync is True


@pytest.mark.parametrize("fail_send", [False, True])
def test_manager_injects_runtime_config_and_cleans_up_only_its_actors(monkeypatch, fail_send):
    import ray

    config = OmegaConf.create(
        {
            "trainer": {"nnodes": 1},
            "actor_rollout_ref": {"rollout": {"engine_kwargs": {"vllm": {"max_model_len": 1024}}}},
        }
    )
    OmegaConf.set_struct(config, True)
    original = OmegaConf.to_container(config)
    received = None
    info = {
        "worker_id": "replica-0-rank-0",
        "zmq_handle": "ipc:///tmp/bootstrap-test",
        "use_shm": True,
        "quant_description": {"quant_method": "ascend", "layer.weight": "W8A8_MXFP8"},
        "manifest": [{"name": "layer.weight", "shape": [64, 32], "dtype": "float8_e4m3fn"}],
    }
    server, worker = object(), object()
    killed = []
    managers = []

    class Manager:
        def __init__(self, config, worker_group, rollout_resource_pool):
            self.config = config
            self.rollout_replicas = [SimpleNamespace(servers=[server])]
            self.ready = False
            managers.append(self)

        async def _initialize_llm_servers(self):
            nonlocal received
            received = asyncio.Event()
            payload = self.config.actor_rollout_ref.rollout.engine_kwargs.vllm.model_loader_extra_config.verl_bootstrap
            await asyncio.to_thread(bootstrap_receiver_ready, payload, info["worker_id"])
            await received.wait()
            await asyncio.to_thread(bootstrap_receiver_loaded, payload, info["worker_id"])
            await asyncio.to_thread(bootstrap_wait_for_release, payload, info["worker_id"])

        async def _init_global_load_balancer(self):
            self.ready = True

    async def send():
        if fail_send:
            raise RuntimeError("test sender failure")
        received.set()

    module = ModuleType("verl.workers.rollout.llm_server")
    module.LLMServerManager = Manager
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append((actor, no_restart)))
    group = SimpleNamespace(
        world_size=1,
        workers=[worker],
        prepare_quantized_weight_sync=lambda: [info],
        bootstrap_quantized_weights=lambda: [send()],
    )
    if fail_send:
        with pytest.raises(RuntimeError, match="test sender failure"):
            create_prequantized_llm_server_manager(config, group, None, 100)
        assert killed == [(server, True), (worker, True)]
        assert not managers[0].ready
    else:
        manager = create_prequantized_llm_server_manager(config, group, None, 100)
        runtime = manager.config.actor_rollout_ref.rollout.engine_kwargs.vllm
        assert runtime.load_format == "verl_prequantized"
        assert runtime.max_model_len == 1024
        assert runtime.model_loader_extra_config.verl_bootstrap.global_steps == 100
        assert runtime.model_loader_extra_config.verl_bootstrap.quant_description == info["quant_description"]
        assert manager.ready
        assert not killed
    assert OmegaConf.to_container(config) == original
