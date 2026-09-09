"""特征掩码相关操作。"""

import torch


def apply_shuffle_replacement_mask(
    inputs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """应用特征掩码，并以批内同列特征替换被丢弃的位置。

    设批内随机排列后的输入为 ``x_shuffled``，则返回
    ``x * mask + (1 - mask) * x_shuffled``。这种方式保持各特征的边际分布，
    避免以常数零作为明显的掩码标记；关于掩码的梯度仍为
    ``x - x_shuffled``。

    Args:
        inputs: 输入张量，形状为 ``[batch_size, input_dim]``。
        mask: 与 ``inputs`` 同形状的软掩码或硬掩码。

    Returns:
        应用替换掩码后的输入张量。
    """
    shuffled_indices = torch.randperm(inputs.size(0), device=inputs.device)
    shuffled_inputs = inputs[shuffled_indices]
    return inputs * mask + (1.0 - mask) * shuffled_inputs
