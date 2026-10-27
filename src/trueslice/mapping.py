"""Accelerate evaluation placement and CPU-backed calibration workers."""
import json
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import (
    convert_file_size_to_int, extract_submodules_state_dict, get_balanced_memory,
    offload_state_dict,
)

from . import utils


_STRATEGIES = ('auto', 'balanced', 'balanced_low_0', 'sequential')


def _parse_device_map(value):
    """Accept a device, an Accelerate strategy, or a JSON module-prefix map."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.lstrip().startswith('{'):
        result = json.loads(value)
        if not isinstance(result, dict):
            raise ValueError('Device map must be a JSON object')
        return result
    return value


def _compute_devices(value=None):
    if value is None or value == 'auto':
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
    if isinstance(value, str):
        values = json.loads(value) if value.lstrip().startswith('[') else value.split(',')
    elif isinstance(value, torch.device):
        values = [value]
    else:
        values = value
    if not isinstance(values, (list, tuple)):
        raise ValueError('cal_device_map must be auto, cpu, a CUDA device, or a list of CUDA devices')
    devices = [torch.device(_device(d.strip() if isinstance(d, str) else d)) for d in values]
    if not devices:
        raise ValueError('Provide at least one calibration compute device')
    devices = [torch.device('cuda:0') if str(d) == 'cuda' else d for d in devices]
    if len(set(devices)) != len(devices):
        raise ValueError('Calibration compute devices must be distinct')
    for device in devices:
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Calibration float64 computations require CPU or CUDA')
        if device.type == 'cuda' and device.index >= torch.cuda.device_count():
            raise ValueError(f'Unavailable calibration device: {device}')
    if len(devices) > 1 and any(d.type != 'cuda' for d in devices):
        raise ValueError('Multiple calibration workers require CUDA devices')
    return devices


def _memory_limits(value):
    """Normalize CUDA names to Accelerate indices without adding memory budgets."""
    if value is None:
        return None
    limits = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(limits, dict) or not limits:
        raise ValueError('max_memory must be a nonempty device-to-memory dictionary')
    normalized = {}
    for device, size in limits.items():
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        elif str(device).startswith('cuda'):
            parsed = torch.device(device)
            device = parsed.index if parsed.index is not None else 0
        if isinstance(device, bool) or not (device == 'cpu' or isinstance(device, int) and device >= 0):
            raise ValueError(f'Unsupported memory budget device: {device}; use cuda:N or cpu')
        if device in normalized:
            raise ValueError(f'Duplicate memory budget for device {device}')
        size = convert_file_size_to_int(size) if isinstance(size, str) else size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('Memory budgets must be positive byte counts or strings such as 9GB')
        normalized[device] = size
    return normalized


def _device(value, *, allow_disk=False):
    if allow_disk and value == 'disk':
        return 'disk'
    if isinstance(value, int) and not isinstance(value, bool):
        value = f'cuda:{value}'
    result = torch.device(value)
    if result.type not in ('cpu', 'cuda'):
        raise ValueError('Mapping devices must be CPU or CUDA')
    return 'cuda:0' if str(result) == 'cuda' else str(result)


def _dispatch_device_map(model, prefixes):
    """Expand a default plus overrides into nonoverlapping Accelerate blocks."""
    result = {}

    def visit(name, module, inherited):
        device = prefixes.get(name, inherited)
        prefix = name + '.' if name else ''
        split = any(key != name and key.startswith(prefix) and value != device
                    for key, value in prefixes.items())
        if not split:
            result[name] = device
            return
        if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
            raise ValueError(f'Evaluation mapping must keep {name or "the root module"} and its direct tensors together')
        for child_name, child in module.named_children():
            visit(prefix + child_name, child, device)

    visit('', model, prefixes[''])
    return result


class Mapping:
    """One placement policy for an experiment.

    Evaluation uses Accelerate. Calibration stores model weights on CPU and
    uses cal_device_map only to select worker compute devices. Each worker keeps
    its activations on its own device. Disk evaluation offload, when needed,
    uses a temporary directory owned by this object; keep it alive during eval.
    """

    def __init__(self, eval_device_map='auto', eval_max_memory=None, cal_device_map='auto'):
        self.compute_devices = _compute_devices(cal_device_map)
        self.eval_max_memory = _memory_limits(eval_max_memory)
        value = _parse_device_map(eval_device_map)
        if isinstance(value, str) and value in _STRATEGIES:
            self.eval_device_map = value
        else:
            values = {'': 'cpu', **value} if isinstance(value, dict) else {'': value}
            self.eval_device_map = {name: _device(device, allow_disk=True) for name, device in values.items()}
            if self.eval_max_memory is not None:
                raise ValueError('eval_max_memory requires an automatic eval_device_map, such as auto')
        self.resolved_eval_device_map = None
        self._offload_directory = None

    def _offload_folder(self):
        if self._offload_directory is None:
            self._offload_directory = TemporaryDirectory(prefix='trueslice-eval-')
        return self._offload_directory.name

    def load(self, model_name, model_path=None, *, evaluate=False, **kwargs):
        """Load directly into evaluation placement, or onto CPU for calibration."""
        from .hf_utils import get_model_and_tokenizer
        # Prefix overrides need the actual module tree before dispatch. Automatic
        # maps and single-device evaluation still load directly to their devices.
        explicit_overrides = isinstance(self.eval_device_map, dict) and len(self.eval_device_map) > 1
        placement = {'device_map': 'cpu'}
        if evaluate and not explicit_overrides:
            placement = dict(device_map=self.eval_device_map, max_memory=self.eval_max_memory,
                             offload_folder=self._offload_folder())
        adapter, tokenizer = get_model_and_tokenizer(model_name, model_path, **kwargs, **placement)
        if evaluate:
            if explicit_overrides:
                self.to_evaluation(adapter)
            else:
                self.resolved_eval_device_map = getattr(adapter.model, 'hf_device_map', None)
        else:
            self.to_calibration(adapter)
        return adapter, tokenizer

    def _resolve_evaluation(self, adapter):
        """Infer a module map using actual parameter sizes, keeping decoders intact."""
        if isinstance(self.eval_device_map, dict):
            names = dict(adapter.model.named_modules())
            for name in self.eval_device_map:
                if name not in names:
                    raise ValueError(f'Unknown mapping module prefix: {name}')
            return _dispatch_device_map(adapter.model, self.eval_device_map)
        else:
            limits = dict(self.eval_max_memory) if self.eval_max_memory is not None else None
            if self.eval_device_map != 'sequential':
                limits = get_balanced_memory(
                    adapter.model, max_memory=limits, no_split_module_classes=adapter.no_split_module_classes,
                    low_zero=self.eval_device_map == 'balanced_low_0')
            return infer_auto_device_map(
                adapter.model, max_memory=limits, no_split_module_classes=adapter.no_split_module_classes)

    @staticmethod
    def input_device(model):
        """Use the embedding execution device, including CPU/disk offload hooks."""
        embedding = model.get_input_embeddings()
        execution = getattr(getattr(embedding, '_hf_hook', None), 'execution_device', None)
        if execution is not None:
            return execution
        if embedding.weight.device.type != 'meta':
            return embedding.weight.device
        return next((device for device in getattr(model, 'hf_device_map', {}).values()
                     if device not in ('cpu', 'disk')), torch.device('cpu'))

    def to_evaluation(self, adapter):
        """Replan evaluation for the current weights, including after slicing."""
        self.to_calibration(adapter)
        device_map = self._resolve_evaluation(adapter)
        disk_modules = [name for name, device in device_map.items() if device == 'disk']
        offload_folder = self._offload_folder() if disk_modules else None
        if disk_modules:
            # Accelerate reuses an existing disk index. Refresh it after fusion
            # or slicing, otherwise evaluation can silently reload stale weights.
            offload_state_dict(offload_folder, extract_submodules_state_dict(adapter.model.state_dict(), disk_modules))
        dispatch_model(adapter.model, device_map=device_map,
                       offload_buffers=True, offload_dir=offload_folder)
        self.resolved_eval_device_map = device_map
        utils.cleanup_memory()

    @staticmethod
    def to_calibration(adapter):
        """Remove evaluation dispatch and store the model on CPU."""
        remove_dispatch(adapter.model)
        adapter.model.cpu()
        utils.cleanup_memory()

    @contextmanager
    def on_compute_device(self, modules, *, device=None):
        """Temporarily place calibration modules, returning them to CPU on exit."""
        device = self.compute_devices[0] if device is None else torch.device(device)
        modules = list(modules)
        try:
            for module in modules:
                module.to(device)
            yield device
        finally:
            for module in modules:
                module.cpu()
            utils.cleanup_memory()


def remove_dispatch(model):
    """Materialize offloaded tensors on CPU before removing inference hooks.

    Detaching directly can restore tensors to their original GPU (or leave meta
    tensors from from_pretrained). Force offload hook restoration to CPU instead.
    """
    def prepare(hook):
        for child in getattr(hook, 'hooks', ()):
            prepare(child)
        if getattr(hook, 'offload', False):
            hook.original_devices = {name: torch.device('cpu') for name in hook.original_devices}

    for module in model.modules():
        if hasattr(module, '_hf_hook'):
            prepare(module._hf_hook)
    remove_hook_from_module(model, recurse=True)
    if any(t.device.type == 'meta' for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('Inference dispatch left unmaterialized tensors; cannot modify model weights')
    model.__dict__.pop('hf_device_map', None)
