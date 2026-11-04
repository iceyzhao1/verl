# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute real lifecycle methods without importing Ray worker or device dependencies.

The AST loader retains the complete production method bodies. Only registration
decorators are removed, so CPU tests can call the methods directly and replace
their external services without starting a training cluster.
"""

import ast
import asyncio
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf


def _load_methods(relative_path, class_name, method_names, namespace):
    source = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names
    ]
    assert {node.name for node in methods} == set(method_names)
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return type("LifecycleHarness", (), {name: namespace[name] for name in method_names})


@pytest.mark.parametrize("bootstrapped", [False, True], ids=["ordinary-startup", "prequantized-startup"])
@pytest.mark.parametrize("resume_mode", ["disable", "auto"], ids=["from-scratch", "restored-checkpoint"])
def test_fit_restores_once_and_validates_the_loaded_step(monkeypatch, tmp_path, bootstrapped, resume_mode):
    events = []
    checkpoint = str(tmp_path / "global_step_100")
    data_path = os.path.join(checkpoint, "data.pt")
    dataloader_state = {"cursor": 100}
    expected_step = 100 if resume_mode == "auto" else 0

    def load_data(path, *, weights_only):
        assert path == data_path
        assert weights_only is False
        events.append("read-data")
        return dataloader_state

    class DataLoader:
        def __len__(self):
            # Step 100 is inside an epoch, so restoring iterator state is required.
            return 12

        def load_state_dict(self, state):
            assert state == dataloader_state
            events.append("restore-data")

    tracking = ModuleType("verl.utils.tracking")
    tracking.Tracking = lambda **kwargs: SimpleNamespace(log=lambda **entry: events.append(("log", entry["step"])))
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    namespace = {
        "os": SimpleNamespace(
            path=SimpleNamespace(join=os.path.join, isabs=os.path.isabs, exists=lambda path: path == data_path),
            getcwd=os.getcwd,
        ),
        "torch": SimpleNamespace(load=load_data),
        "Role": SimpleNamespace(Critic="critic"),
        "find_latest_ckpt_path": lambda folder: checkpoint,
        "pprint": lambda message: None,
    }
    harness = _load_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", ["_load_checkpoint", "fit"], namespace)
    trainer = harness()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "lifecycle-test",
                "experiment_name": "startup",
                "logger": [],
                "val_before_train": True,
                "val_only": True,
                "resume_mode": resume_mode,
                "default_hdfs_dir": None,
                "default_local_dir": str(tmp_path),
                "del_local_ckpt_after_load": False,
            }
        }
    )
    trainer.global_steps = 0
    trainer.use_critic = True

    def load_worker(role, path, *, del_local_after_load):
        assert path == os.path.join(checkpoint, role)
        assert del_local_after_load is False
        events.append(f"load-{role}")

    trainer.actor_rollout_wg = SimpleNamespace(
        load_checkpoint=lambda *args, **kwargs: load_worker("actor", *args, **kwargs)
    )
    trainer.critic_wg = SimpleNamespace(load_checkpoint=lambda *args, **kwargs: load_worker("critic", *args, **kwargs))
    trainer.train_dataloader = DataLoader()
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._shutdown_dump_executor = lambda: events.append("shutdown")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append(("sync", step)))

    def validate():
        events.append(("validate", trainer.global_steps))
        return {"validation/test": 1}

    trainer._validate = validate
    trainer._quantized_startup_done = bootstrapped
    if bootstrapped:
        # init_workers restores before creating the inference servers. Exercise
        # that restoration and then fit's handling of the completed bootstrap.
        trainer._load_checkpoint()
        events.append(("bootstrap", trainer.global_steps))

    trainer.fit()

    expected = ["load-actor", "load-critic", "read-data", "restore-data"] if resume_mode == "auto" else []
    expected.extend(
        [
            ("bootstrap" if bootstrapped else "sync", expected_step),
            ("validate", expected_step),
            ("log", expected_step),
            "shutdown",
        ]
    )
    assert trainer.global_steps == expected_step
    assert events == expected


@pytest.mark.parametrize(
    "failure", [None, "export", "send", "cancel"], ids=["success", "export-error", "send-error", "cancelled"]
)
def test_bootstrap_releases_training_memory_on_every_exit(monkeypatch, failure):
    events = []
    weights = SimpleNamespace(close=lambda: events.append("close-generator"))

    def export(*, prequantized):
        assert prequantized is True
        events.append("export")
        if failure == "export":
            raise RuntimeError("export failed")
        return weights, None

    class Sender:
        def __init__(self, *, zmq_handle, bucket_size_mb, use_shm):
            assert zmq_handle == "ipc:///bootstrap-test"
            assert bucket_size_mb == 1
            assert use_shm is True

        async def async_send_weights(self, received_weights):
            assert received_weights is weights
            events.append("send")
            if failure == "send":
                raise RuntimeError("send failed")
            if failure == "cancel":
                raise asyncio.CancelledError

    transfer = ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    transfer.BucketedWeightSender = Sender
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)

    def empty_cache(*, force_sync):
        assert force_sync is True
        events.append("empty-cache")

    harness = _load_methods(
        "verl/workers/engine_workers.py",
        "ActorRolloutRefWorker",
        ["bootstrap_quantized_weights"],
        {
            "set_expandable_segments": lambda enabled: events.append(("expand", enabled)),
            "aggressive_empty_cache": empty_cache,
        },
    )
    worker = harness()
    worker.config = OmegaConf.create(
        {"rollout": {"quantized_weight_sync": True, "checkpoint_engine": {"update_weights_bucket_megabytes": 1}}}
    )

    def offload(device, *, model, optimizer, grad):
        assert (device, model, optimizer, grad) == ("cpu", True, False, False)
        events.append("offload")

    worker.actor = SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=export), to=offload)
    worker.rollout = SimpleNamespace(zmq_handle="ipc:///bootstrap-test", use_shm=True)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(worker.bootstrap_quantized_weights())
    elif failure is not None:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            asyncio.run(worker.bootstrap_quantized_weights())
    else:
        asyncio.run(worker.bootstrap_quantized_weights())

    expected = [("expand", False), "export"]
    if failure != "export":
        expected.extend(["send", "close-generator"])
    expected.extend(["offload", "empty-cache", ("expand", True)])
    assert events == expected
