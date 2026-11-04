# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Coordinate only the first trainer-to-Ascend parameter load."""

import asyncio
from copy import deepcopy

from omegaconf import open_dict

from .bootstrap_sync import BootstrapCoordinator

BOOTSTRAP_TIMEOUT_SECONDS = 600.0


def build_bootstrap_payload(worker_infos, coordinator_config, global_steps):
    """Require one identical export contract and one unique endpoint per training rank."""
    if not worker_infos:
        raise ValueError("quantized_weight_sync requires training workers")
    reference = worker_infos[0]
    endpoints = {}
    for info in worker_infos:
        for key in ("quant_description", "manifest", "use_shm"):
            if info[key] != reference[key]:
                raise ValueError(f"Training workers disagree on {key}")
        worker_id = info["worker_id"]
        if worker_id in endpoints or info["zmq_handle"] in endpoints.values():
            raise ValueError(f"Duplicate bootstrap endpoint: {worker_id}")
        endpoints[worker_id] = info["zmq_handle"]
    return {
        **coordinator_config,
        "data_endpoints": endpoints,
        "use_shm": reference["use_shm"],
        "manifest": reference["manifest"],
        "quant_description": reference["quant_description"],
        "global_steps": global_steps,
    }


async def initialize_with_bootstrap(manager, coordinator, send_weights, abort_workers):
    """Run initialization and sending concurrently; any failure fails the whole startup."""

    async def send_when_ready():
        await asyncio.to_thread(coordinator.wait_ready)
        # The sender RPC returns only after its generator and device temporaries are released.
        await asyncio.wait_for(asyncio.gather(*send_weights()), timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        await asyncio.to_thread(coordinator.wait_loaded)
        coordinator.release()

    initialize = asyncio.create_task(manager._initialize_llm_servers())
    send = asyncio.create_task(send_when_ready())
    try:
        await asyncio.gather(initialize, send)
        await manager._init_global_load_balancer()
        return manager
    except BaseException as error:
        coordinator.abort(str(error))
        # A failed bucket operation can block in the unchanged transport. Tear down
        # this failed startup's actors rather than pretending that it can be resumed.
        abort_workers()
        initialize.cancel()
        send.cancel()
        await asyncio.gather(initialize, send, return_exceptions=True)
        raise
    finally:
        coordinator.close()


def create_prequantized_llm_server_manager(config, worker_group, rollout_resource_pool, global_steps):
    """Keep the existing server manager and data channel; add the startup rendezvous."""
    import ray

    from verl.workers.rollout.llm_server import LLMServerManager

    if config.trainer.nnodes != 1:
        raise ValueError("quantized_weight_sync currently supports the single-node HYBRID deployment")
    infos = worker_group.prepare_quantized_weight_sync()
    if len(infos) != worker_group.world_size:
        raise ValueError("Every FSDP rank must participate in quantized_weight_sync")
    host = ray.util.get_node_ip_address()
    coordinator = BootstrapCoordinator(
        [info["worker_id"] for info in infos], host=host, timeout_seconds=BOOTSTRAP_TIMEOUT_SECONDS
    )
    try:
        payload = build_bootstrap_payload(infos, coordinator.serializable_config(), global_steps)
        runtime_config = deepcopy(config)
        with open_dict(runtime_config.actor_rollout_ref.rollout):
            engine_kwargs = runtime_config.actor_rollout_ref.rollout.engine_kwargs.vllm
            extra = dict(engine_kwargs.get("model_loader_extra_config") or {})
            if "verl_bootstrap" in extra:
                raise ValueError("verl_bootstrap is generated internally; do not supply it in engine_kwargs")
            extra["verl_bootstrap"] = payload
            engine_kwargs.model_loader_extra_config = extra
            engine_kwargs.load_format = "verl_prequantized"
        manager = LLMServerManager(
            config=runtime_config, worker_group=worker_group, rollout_resource_pool=rollout_resource_pool
        )
    except BaseException:
        coordinator.close()
        raise

    def abort_workers():
        servers = [server for replica in getattr(manager, "rollout_replicas", []) for server in replica.servers]
        for actor in servers + list(worker_group.workers):
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass  # Preserve the original startup exception if an actor already exited.

    return asyncio.run(
        initialize_with_bootstrap(manager, coordinator, worker_group.bootstrap_quantized_weights, abort_workers)
    )
