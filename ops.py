import torch
import torch.nn.functional as F


def norm_loss(predictions, labels, mask=None, loss_type="l1", reduce_mean=True, p=1):
    """
    compatible with tf1
    detail tf: https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    detail torch l1: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
    detail torch l2: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
    """
    if mask is not None:
        predictions = predictions * mask
        labels = labels * mask

    if loss_type == "l1":
        loss = F.l1_loss(predictions, labels, reduction='sum')
    elif loss_type == "l2":
        loss = F.mse_loss(predictions, labels, reduction='sum') / 2

    elif loss_type == "l2,1":
        loss = torch.sqrt(torch.sum(torch.square(predictions - labels) + 1e-16, dim=1))
        if p != 1:
            loss = torch.pow(loss, p)
        return torch.sum(loss)
    else:
        loss = 0
        assert "available: 'l1', 'l2', 'l2,1'"

    if reduce_mean:
        numel = torch.prod(torch.tensor(predictions.shape)).float()
        loss = torch.div(loss, numel)

    return loss


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    batch_size, height, width = x.shape
    batch_idx = torch.arange(0, batch_size).view((batch_size, 1, 1)).type(torch.int64)
    b = batch_idx.repeat(1, height, width)

    value = img[b.long(), :, y.long(), x.long()]
    return value
