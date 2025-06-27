> 🧪 **Note**: This repository is just [NNFS](https://nnfs.io) with added convolution layers + max pooling and basic Le-net architecture implemented for educational purposes. 

### Enhancements

#### Convolutions

> ⚠️ **Important Notice: `Conv3D` is incorrectly implemented.**

In this codebase, `Conv2D` and `Conv3D` layers are implemented as:
- `Conv2D`: input shape `[m, h, w]`
- `Conv3D`: input shape `[m, h, w, nc]`

However, this diverges from how PyTorch handle convolutional layers:

- **PyTorch uses `Conv2d` for both grayscale and RGB inputs.**
- `[batch, channels, height, width]` .
