import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import CSRNet_vgg_13 as vgg
import numpy as np
def back_end(vgg):
    num_filter=[512,512,512,256,128,64]
    conv=vgg
    for i in range(6):
         conv= fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter[i],
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',dilation=1,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=0.01)),
                bias_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.0)))
    return conv
def CSRNet(input,size):
    input=fluid.layers.reshape(input,[-1,3,size[1],size[0]])
    vgg_13=vgg.VGG13(input)
    conv=fluid.layers.conv2d(
                input=back_end(vgg_13),
                num_filters=1,
                filter_size=1,
                stride=1,
                padding=0,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=0.01)),
                bias_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.0)))
    return fluid.layers.image_resize(input=conv,out_shape=[size[1],size[0]])
