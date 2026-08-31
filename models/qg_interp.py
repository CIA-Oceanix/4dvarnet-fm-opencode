import torch


def spectral_resize_2d(field: torch.Tensor, ny_out: int, nx_out: int,
                       device: torch.device | None = None) -> torch.Tensor:
    """Spectral resize (upsample/downsample) of a real field on the last two dims.

    Zero-pads (upsample) or truncates (downsample) the field in wavenumber
    (Fourier) space, preserving the band-limited spectral content. Works on
    arbitrary leading batch dims; the last two dims are (ny, nx).

    Args:
        field: (..., ny_in, nx_in) real tensor.
        ny_out, nx_out: target spatial resolution (even).
        device: optional target device; defaults to the input's device.

    Returns:
        (..., ny_out, nx_out) real tensor.
    """
    if device is not None:
        field = field.to(device)
    ny_in, nx_in = field.shape[-2], field.shape[-1]
    if ny_in == ny_out and nx_in == nx_out:
        return field
    fh = torch.fft.rfft2(field, dim=(-2, -1))
    fh_out = torch.zeros(
        (*fh.shape[:-2], ny_out, nx_out // 2 + 1),
        dtype=fh.dtype, device=fh.device,
    )
    ny_min = min(ny_in, ny_out)
    # rfft holds non-negative x-wavenumbers only, length nx_in//2+1.
    nk_in = nx_in // 2 + 1
    nk_out = nx_out // 2 + 1
    nk = min(nk_in, nk_out)
    fh_out[..., : ny_min // 2, :nk] = fh[..., : ny_min // 2, :nk]
    fh_out[..., -ny_min // 2:, :nk] = fh[..., -ny_min // 2:, :nk]
    # amplitude scaling preserves physical magnitude under resolution change.
    scale = float(ny_out * nx_out) / float(ny_in * nx_in)
    fh_out = fh_out * scale
    return torch.fft.irfft2(fh_out, s=(ny_out, nx_out), dim=(-2, -1))
