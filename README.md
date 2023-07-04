### Neural Networks from Scratch (NNFS) book code

This repository is contains the code from the book NNFS([link](https://nnfs.io/)) with  arranged the chapter lectures into packages so that its easier to use the code by just importing them.

####Enhancement

#### Convolutions:
Convolutions classes are implemented with shapes conv2d[m,h,w] and conv3d[m,h,w,nc] current suported parameters are padding(zero-padding) and strides,weights parameters takes a tupple as input with shape 2d[h,w,nc_out] and 3d[h,w,nc_in,nc_out],biases int for shape that should match nc_out from weights.

Tests are written for zero-padding(2d,3d) and conv2d,conv3d(forward,backward) and its performed by initializing inputs/parameters in pytorch performing the forward/backward in pytorch and our conv implementation with same params,and comparing the values with np.allclose() because of numerical instability.
     
code example conv3d:
```python
inputs=np.random.randn(5,10,10,3) 
conv3d=Conv3D([3,3,3,6],6,padding=1,stride=1)
conv3d.forward(inputs,training=True)
dvalues=np.ones_like(conv3d.output)
conv3d.backward(dvalues)
```

#### Max pooling:
Currently maxpool(2d,3d) are fully implemented with parameters f(filter size) and stride,if we take an input[m,n_h_prev,n_w_prev,nc] the output is gonna be [m,nh,nw,nc],nh/nw are calcuated 
((n_h_prev-f)/stride)+1

Test are written the same way as conv tests,by performing forward/backward pass in pytorch and our maxpool layers with same input/params and comparing the output values and grad values with np.allclose()

code example maxpool3d:
```python
inputs=np.random.randn(1,3,3,1)
pool3d=MaxPooling3D(2,stride=1)
pool3d.forward(input_permuted,training=True)
dvalues=np.ones_like(input_permuted)
pool3d.backward(dvalues)
```

#### Models:
Mnist via cnn's is done and is mostly inspired by LeNet 5 architecture except we modified the architecture to use max-pool instead of average-pool and using Relu as activation function.

Cifar10 is done are currently working on finding the optimal solution for hyper-parameters to make the network train faster with less resources/layers.

     


           
   





            







 









        






 





