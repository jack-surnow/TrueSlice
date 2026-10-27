"""Persistent calibration workers, with a separate spawned process per GPU."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import gc
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from .model_utils import get_signals
from .utils import map_tensors


class _Worker:
    def __init__(self, device):
        self.device = torch.device(device)
        self.args = self.kwargs = self.masks = self.basis = None

    def run(self, method, *args):
        context = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
        with torch.no_grad(), context:
            try:
                return getattr(self, method)(*args)
            finally:
                if self.device.type == 'cuda' and method in ('profile', 'rotate'):
                    # Release temporary weights/workspaces while retaining this
                    # worker's live calibration activations.
                    torch.cuda.empty_cache()

    def initialize(self, args, kwargs, masks):
        self.args, self.kwargs, self.masks = map_tensors(
            (args, kwargs, masks), device=self.device)

    def profile(self, layer, index, parallel):
        from .rotate import get_scatter_mat

        try:
            layer.layer.to(self.device)
            moments = None
            for i, args in enumerate(self.args):
                groups = get_signals(
                    layer, [args], [self.kwargs[i]['layer_kwargs'][index]],
                    device=self.device, storage_device=self.device, capture_mlp_input=not parallel,
                )
                masks = [self.masks[i]] if self.masks else []
                batch_moments = [get_scatter_mat(group, masks, device=self.device) for group in groups]
                if moments is None:
                    moments = batch_moments
                else:
                    for total, contribution in zip(moments, batch_moments):
                        total.add_(contribution)
                # Replace each batch promptly: keep neither the old activation
                # shard nor all intermediate MLP inputs alongside the outputs.
                self.args[i] = layer.get_updated_args(groups[-1][0], args)
                del groups, batch_moments, args
            return [moment.cpu() for moment in moments]
        finally:
            # Executor requests remain alive until their result has been sent.
            # Release GPU weights before the parent resumes or handles an error.
            layer.layer.cpu()

    def finish_calibration(self):
        self.args = self.kwargs = self.masks = None
        gc.collect()
        if self.device.type == 'cuda':
            # Release cached layer/activation allocations before the parent solves
            # the eigensystem on the first GPU.
            torch.cuda.empty_cache()

    def set_basis(self, basis):
        self.basis = basis.to(self.device)

    def rotate(self, layer, width):
        from .rotate import (
            rotate_attention_inputs, rotate_attention_output, rotate_mlp_input, rotate_mlp_output,
            slice_attention_inputs, slice_attention_output, slice_mlp_input, slice_mlp_output,
        )

        rotate_attention_inputs(layer, self.basis)
        slice_attention_inputs(layer, width)
        rotate_attention_output(layer, self.basis)
        slice_attention_output(layer, width)
        rotate_mlp_input(layer, self.basis)
        slice_mlp_input(layer, width)
        rotate_mlp_output(layer, self.basis)
        slice_mlp_output(layer, width)
        return layer


_worker = None


def _initialize_worker(device, seed, num_threads, dtype):
    global _worker
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(num_threads)
    torch.set_default_dtype(dtype)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    _worker = _Worker(device)


def _run_worker(method, args):
    return _worker.run(method, *args)


class ComputeWorkers:
    """One serial executor per device keeps kernels and calibration data isolated.

    CPU tensors are transported through PyTorch's shared-memory reducers. Keep
    requests and replies on CPU; each worker retains activations on its compute
    device across layers. A single device runs directly in the caller.
    """

    def __init__(self, devices):
        self.devices = devices
        self.executors = []
        self.local = None

    def __enter__(self):
        if len(self.devices) == 1:
            self.local = _Worker(self.devices[0])
        else:
            context = mp.get_context('spawn')
            num_threads = max(1, torch.get_num_threads() // len(self.devices))
            try:
                for index, device in enumerate(self.devices):
                    self.executors.append(ProcessPoolExecutor(
                        max_workers=1, mp_context=context, initializer=_initialize_worker,
                        initargs=(device, (torch.initial_seed() + index) % (2**64),
                                  num_threads, torch.get_default_dtype()),
                    ))
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def call(self, method, jobs):
        """Run job i on device i and return results in the same order."""
        if len(jobs) > len(self.devices):
            raise ValueError('Provide at most one job per compute device')
        if self.local is not None:
            return [self.local.run(method, *args) for args in jobs]
        futures = [executor.submit(_run_worker, method, args)
                   for executor, args in zip(self.executors, jobs)]
        results = []
        for device, future in zip(self.devices, futures):
            try:
                results.append(future.result())
            except Exception as error:
                error.add_note(f'TrueSlice worker on {device} failed during {method}')
                raise
        return results

    def __exit__(self, *_):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self.executors.clear()
        self.local = None
