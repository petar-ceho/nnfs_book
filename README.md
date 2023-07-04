### Neural Networks from Scratch (NNFS) book code

This repository is contains the code from the book NNFS([link](https://nnfs.io/)) with  arranged the chapter lectures into packages so that its easier to use the code by just importing them.


#### Convolutions:
Convolutions classes are implemented with shapes conv2d[m,h,w] and conv3d[m,h,w,nc] current suported parameters are padding(zero-padding) and strides,weights parameters takes a tupple as input with shape 2d[h,w,nc_out] and 3d[h,w,nc_in,nc_out],biases int for shape that should match nc_out from weights.

Tests are written for zero-padding(2d,3d) and conv2d,conv3d(forward,backward) and its performed by initializing inputs/parameters in pytorch performing the forward/backward in pytorch and our conv implementation with same params,and comparing the values with np.allclose() because of numerical instability.
     
code example conv3d:
```python
conv3d=Conv3D([3,3,3,6],6,padding=1,stride=1)
conv3d.set_params(weights,biases)
conv3d.forward(inputs,training=True)
dvalues=np.ones_like(conv3d.output)
conv3d.backward(dvalues)
```








 









        






 





