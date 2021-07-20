"""Functions for sampling before the dataloader is applied, many of these have been adapted from spert repo
https://github.com/lavis-nlp/spert/blob/master/spert/sampling.py
"""
from typing import List

import torch

TO_PRESERVE = ['input_ids', 'token_type_ids', 'attention_mask', 'overflow_to_sample_mapping', 'labels']
DYPES_MAPPING = {
    'entity_masks': torch.bool,
    'rel_tuples': torch.int64,
    'ctx_mask': torch.bool,
    'rel_labels': torch.int64,
    'ctx_len': torch.int64
}


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key in TO_PRESERVE:
            padded_batch[key] = torch.stack([s[key] for s in batch])
        else:
            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                padded_batch[key] = padded_stack([s[key] for s in batch], inp_key=key)

    return padded_batch


def padded_stack(tensors, inp_key, padding=0):
    dim_count = max([len(t.shape) for t in tensors])

    max_shape = [max([t.shape[d] for t in tensors if len(t.shape) == dim_count]) for d in range(dim_count)]
    padded_tensors = []
    tensors_dtype = DYPES_MAPPING[inp_key]
    # tensors_dtype = get_tensors_dtype(inp_tensors=tensors)
    # print(check_tensors_dtypes)
    for t in tensors:
        e = extend_tensor(t, max_shape, tensors_dtype=tensors_dtype, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, tensors_dtype, fill=0):
    tensor_shape = tensor.shape
    extended_tensor = torch.zeros(extended_shape, dtype=tensors_dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)
    if not len(tensor_shape) < len(extended_shape):
        if len(tensor_shape) == 1:
            extended_tensor[:tensor_shape[0]] = tensor
        elif len(tensor_shape) == 2:
            extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
        elif len(tensor_shape) == 3:
            extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
        elif len(tensor_shape) == 4:
            extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor
    return extended_tensor


def get_tensors_dtype(inp_tensors: List[torch.Tensor]):
    dtypes = set([t.dtype for t in inp_tensors])
    if len(dtypes) > 1:
        main_dtype = None
        if torch.bool in dtypes:
            main_dtype = torch.bool
        if torch.int64 in dtypes:
            main_dtype = torch.int64
        if not main_dtype:
            raise ValueError
    else:
        main_dtype = list(dtypes)[0]
    return main_dtype


def batch_index(tensor, index):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])


