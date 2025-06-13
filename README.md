> 🧪 **Note**: This is part of an experimental extension to the original [NNFS](https://nnfs.io) book code, maintained as a personal fork.

### Enhancements

#### Convolutions

> ⚠️ **Important Notice: `Conv3D` is incorrectly implemented.**

In this codebase, `Conv2D` and `Conv3D` layers are implemented as:
- `Conv2D`: input shape `[m, h, w]`
- `Conv3D`: input shape `[m, h, w, nc]`

However, this diverges from how PyTorch (and most standard deep learning libraries) handle convolutional layers:

- **PyTorch uses `Conv2d` for both grayscale and RGB inputs.**
  - It expects input shape `[batch, channels, height, width]` regardless of the number of channels.
  - Therefore, even for RGB (3-channel) images, `Conv2d` is used, **not** `Conv3d`.

- **`Conv3d` in PyTorch is intended for volumetric data**, such as video or 3D medical scans, with input shape `[batch, channels, depth, height, width]`.

This means the current `Conv3D` implementation in this codebase is misleadingly named and structured—it's essentially a `Conv2D` layer that accepts multi-channel input, not a true 3D convolution layer.

##### Example usage of current (non-standard) `Conv3D`:
```python
inputs = np.random.randn(5, 10, 10, 3)  # shape: [batch, height, width, channels]
conv3d = Conv3D([3, 3, 3, 6], 6, padding=1, stride=1)
conv3d.forward(inputs, training=True)
dvalues = np.ones_like(conv3d.output)
conv3d.backward(dvalues)
