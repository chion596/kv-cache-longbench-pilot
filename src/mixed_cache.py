"""Разная битность K/V в stock HQQ и подсчёт уникальных storages cache."""
from __future__ import annotations

from dataclasses import dataclass
import time

from src.run_quality import eos_ids_for


@dataclass
class ExactTensor:
    tensor: object


def policy_bits(policy):
    if policy == 'dynamic':
        return 16, 16
    import re
    match = re.fullmatch(r'k(4|8|16)_v(4|8|16)', policy)
    if not match:
        raise ValueError(f'Неизвестная политика: {policy}')
    return tuple(map(int, match.groups()))


def make_mixed_cache(model, policy, residual=512):
    from transformers import DynamicCache, QuantizedCache
    bits = policy_bits(policy)
    if policy == 'dynamic':
        cache = DynamicCache(config=model.config)
        return cache, [[0, 0] for _ in cache.layers]
    cache = QuantizedCache(backend='hqq', config=model.config, nbits=4,
        axis_key=1, axis_value=1, q_group_size=64, residual_length=residual)
    counts = [[0, 0] for _ in cache.layers]
    for index, layer in enumerate(cache.layers):
        state = {'slot': 0}
        quantize, dequantize = layer._quantize, layer._dequantize

        def encode(tensor, axis, *, layer=layer, quantize=quantize, index=index, state=state):
            slot = state['slot']
            if slot >= 2:
                raise ValueError('Неожиданный refresh в mixed_precision_v1')
            state['slot'] += 1
            if bits[slot] == 16:
                return ExactTensor(tensor.clone())
            old = layer.nbits
            try:
                layer.nbits = bits[slot]
                packed = quantize(tensor, axis=axis)
                if packed[1]['nbits'] != bits[slot]:
                    raise ValueError('HQQ сохранил другую битность')
                counts[index][slot] += 1
                return packed
            finally:
                layer.nbits = old

        def decode(packed, *, dequantize=dequantize):
            return packed.tensor if isinstance(packed, ExactTensor) else dequantize(packed)

        layer._quantize, layer._dequantize = encode, decode
    return cache, counts


def storage_bytes(objects):
    """Считаем весь удерживаемый storage один раз, включая общие views/metadata."""
    import torch
    seen, by_device = set(), {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            size = storage.nbytes()
            key = str(value.device), storage.data_ptr()
            if size and key not in seen:
                seen.add(key)
                by_device[key[0]] = by_device.get(key[0], 0)+size
        elif isinstance(value, ExactTensor):
            visit(value.tensor)
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                visit(v)

    visit(objects)
    return {'cache_device_bytes': by_device, 'cache_storage_count': len(seen),
        'cache_gpu_bytes_after_query': sum(v for k, v in by_device.items() if k.startswith('cuda')),
        'cache_host_bytes_after_query': sum(v for k, v in by_device.items() if k.startswith('cpu'))}


def cache_snapshot(cache):
    objects, bits = [], []
    for layer in cache.layers:
        objects.extend([layer.keys, layer.values])
        if hasattr(layer, '_quantized_keys'):
            pair = [layer._quantized_keys, layer._quantized_values]
            objects.extend(pair)
            bits.append([p.tensor.element_size()*8 if isinstance(p, ExactTensor) else int(p[1]['nbits']) for p in pair])
        else:
            bits.append([layer.keys.element_size()*8, layer.values.element_size()*8])
    return {**storage_bytes(objects), 'stored_bits_by_layer': bits,
            'cache_tokens_after_query': cache.get_seq_length()}


def native_generate(model, tokenizer, task, policy, *, residual=512, trace=False, factory=None):
    """Native split/greedy; snapshot после вопроса, перед первым decode-шагом."""
    import torch
    device = model.get_input_embeddings().weight.device
    sync = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    ids = task['input_ids']
    split = task['context_split']
    limit = task['max_new_tokens']
    if not 0 < split < len(ids) or limit < 1 or len(ids)-split+limit >= residual:
        raise ValueError('Некорректная граница/лимит или риск refresh')
    cache, counts = make_mixed_cache(model, policy, residual) if factory is None else factory(model)
    attention = torch.ones((1, len(ids)), dtype=torch.long, device=device)
    tensor = torch.tensor([ids], device=device)
    positions = torch.arange(len(ids), device=device)
    traces = []

    def logits(output):
        result = output.logits[:, -1, :]
        if trace:
            traces.append(result.detach().float().cpu())
        return result

    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=tensor[:, :split], attention_mask=attention[:, :split],
            cache_position=positions[:split], past_key_values=cache, use_cache=True, logits_to_keep=1)
        del output
        output = model(input_ids=tensor[:, split:], attention_mask=attention,
            cache_position=positions[split:], past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = logits(output).argmax(-1, keepdim=True)
        generated = [int(token.item())]
        del output
        sync()
        memory = cache_snapshot(cache)
        prefill_done = time.perf_counter()
        eos = eos_ids_for(tokenizer, model)
        position = torch.tensor([len(ids)], device=device)
        for _ in range(limit-1):
            if generated[-1] in eos:
                break
            attention = torch.cat([attention, attention.new_ones((1, 1))], dim=-1)
            output = model(input_ids=token, attention_mask=attention, cache_position=position,
                past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = logits(output).argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            position += 1
            del output
        sync()
        ended = time.perf_counter()
    stats = {**memory, 'prefill_sec': prefill_done-started, 'decode_sec': ended-prefill_done,
        'output_ids': generated, 'output_tokens': len(generated),
        'stop_reason': 'eos' if generated[-1] in eos else 'max_new_tokens',
        'hqq_calls_by_layer': counts}
    return tokenizer.decode(generated, skip_special_tokens=True), stats, traces


def assert_parity(stats, trace, other_stats, other_trace):
    import torch
    if (stats['output_ids'] != other_stats['output_ids'] or len(trace) != len(other_trace)
            or any(not torch.equal(a, b) for a, b in zip(trace, other_trace))):
        raise ValueError('Native logits/output IDs не совпали со stock/exact cache')
    return True
