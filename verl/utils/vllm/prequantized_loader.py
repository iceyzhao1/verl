# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""vLLM loader plugin for first weights supplied by the colocated trainer."""

import os

_registered = False


def register_prequantized_loader():
    """Register in every vLLM worker; never start communications on import."""
    global _registered
    if not _registered:
        from vllm.model_executor.model_loader import register_model_loader
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        class VerlPrequantizedModelLoader(BaseModelLoader):
            def download_model(self, model_config):
                # Architecture/config/tokenizer still use the HF model path.
                # Only checkpoint tensor loading is replaced by this loader.
                pass

            def load_weights(self, model, model_config):
                _load_first_weights(self.load_config, model)

        register_model_loader("verl_prequantized")(VerlPrequantizedModelLoader)
        _registered = True


def _load_first_weights(load_config, model):
    import torch
    from vllm.config import get_current_vllm_config

    from verl.utils.vllm.bootstrap_sync import (
        bootstrap_receiver_failed,
        bootstrap_receiver_loaded,
        bootstrap_receiver_ready,
        bootstrap_wait_for_release,
    )
    from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
    from verl.utils.vllm.prequantized_weights import (
        PrequantizedWeightLoad,
        bootstrap_config_from_load_config,
    )
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver
    from verl.workers.rollout.vllm_rollout.utils import _resolve_vllm_weight_sync_local_rank

    config = bootstrap_config_from_load_config(load_config)
    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    # This mode is deliberately scoped to one node per replica. The caller
    # validates that topology before starting any model subprocesses.
    local_rank = _resolve_vllm_weight_sync_local_rank(parallel_config.rank, parallel_config)
    worker_id = f"replica-{os.environ['VERL_REPLICA_RANK']}-rank-{local_rank}"
    try:
        endpoint = config["data_endpoints"][worker_id]
        device = next(model.parameters()).device
        if device.index is None and device.type != "cpu":
            from verl.utils.device import get_device_id

            device = torch.device(device.type, get_device_id())
        patch_vllm_moe_model_weight_loader(model)
        load = PrequantizedWeightLoad(model, config["manifest"], initial_load=True)
        receiver = BucketedWeightReceiver(endpoint, device=device, use_shm=config.get("use_shm", False))
        bootstrap_receiver_ready(config, worker_id)
        receiver.receive_weights(load.load_bucket)
        load.finish()
        # receive_weights has now synchronized and released the IPC/SHM
        # buffer. BaseModelLoader runs common postprocessing after return.
        bootstrap_receiver_loaded(config, worker_id)
        bootstrap_wait_for_release(config, worker_id)
    except Exception as error:
        try:
            bootstrap_receiver_failed(config, worker_id, error)
        except Exception:
            pass  # Preserve the original loader/transport failure.
        raise
