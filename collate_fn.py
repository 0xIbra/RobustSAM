import torch

def custom_collate(batch):
    clear_im, degraded_im, clear_path, all_mask, input_point, input_label = zip(*batch)

    # Stack image tensors
    clear_im = torch.stack(clear_im, 0)
    degraded_im = torch.stack(degraded_im, 0)

    # Handle all_mask with padding
    max_masks = max(mask.shape[0] for mask in all_mask)
    masks_padded = []
    for mask in all_mask:
        if mask.shape[0] < max_masks:
            pad_size = max_masks - mask.shape[0]
            pad = torch.zeros((pad_size, mask.shape[1], mask.shape[2]), dtype=mask.dtype)
            masks_padded.append(torch.cat([mask, pad], 0))
        else:
            masks_padded.append(mask[:max_masks])
    all_mask = torch.stack(masks_padded, 0)

    # Stack input points and labels
    input_point = torch.stack(input_point, 0)
    input_label = torch.stack(input_label, 0)

    return clear_im, degraded_im, clear_path, all_mask, input_point, input_label
