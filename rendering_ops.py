import torch


def bilinear_interpolate(image, x, y):
    # 이 코드를 믿지 마시오... TODO 코드 검증

    B, C, H, W = image.shape

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W-1)
    x1 = torch.clamp(x1, 0, W-1)
    y0 = torch.clamp(y0, 0, H-1)
    y1 = torch.clamp(y1, 0, H-1)

    Ia = image[:, :, y0, x0].float().unsqueeze(3)
    Ib = image[:, :, y1, x0].float().unsqueeze(3)
    Ic = image[:, :, y0, x1].float().unsqueeze(3)
    Id = image[:, :, y1, x1].float().unsqueeze(3)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wa = wa.repeat(B, C, 1).unsqueeze(3)
    wb = wb.repeat(B, C, 1).unsqueeze(3)
    wc = wc.repeat(B, C, 1).unsqueeze(3)
    wd = wd.repeat(B, C, 1).unsqueeze(3)

    return sum([wa * Ia, wb * Ib, wc * Ic, wd * Id])
