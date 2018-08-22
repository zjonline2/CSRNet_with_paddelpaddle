import paddle
import paddle.fluid as fluid

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


def net(input,layers):
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 3, 3]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }

        nums = vgg_spec[layers]
        conv1 = conv_block(input, 64, nums[0])
        print conv1
        conv1=fluid.layers.pool2d(
            input=conv1, pool_size=2, pool_type='max', pool_stride=2)
        print conv1
        conv2 = conv_block(conv1, 128, nums[1])
        print conv2
        conv2=fluid.layers.pool2d(
            input=conv2, pool_size=2, pool_type='max', pool_stride=2)
        print conv2
        conv3 = conv_block(conv2, 256, nums[2])
        print conv3
        conv3=fluid.layers.pool2d(
            input=conv3, pool_size=2, pool_type='max', pool_stride=2)
        print conv3
        conv4 = conv_block(conv3, 512, nums[3])
        print conv4
        return conv4

def conv_block(input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=0.01)),
                bias_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.0)))
        return conv
def VGG13(input):
    model = net(input,13)
    return model

