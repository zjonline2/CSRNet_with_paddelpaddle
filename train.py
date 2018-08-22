import os
import time
import numpy as np
import argparse
import functools
import shutil
import CSRNet as net
import paddle
import paddle.fluid as fluid
import reader
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.001,     "Learning rate.")
add_arg('batch_size',       int,   1,        "Minibatch size.")
add_arg('num_passes',       int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Parallel.")
add_arg('dataset',          str,   'baidu_star_2018', "dataset")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   '', "The init model path.")
add_arg('apply_distort',    bool,  False,   "Whether apply distort.")
add_arg('apply_expand',     bool,  False,  "Whether appley expand.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('ap_version',       str,   '11point',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.") 
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")
add_arg('is_toy',           int,   0, "Toy for quick debug, 0 means using all data, while n means using only n sample.")
add_arg('for_model_ce',     bool,  False, "Use CE to evaluate the model")
add_arg('data_dir',         str,   './../baidu_star_2018', "data directory")
add_arg('skip_batch_num',   int,    5,  "the num of minibatch to skip.")
add_arg('iterations',       int,   120,  "mini batchs.")
add_arg('with_memory_optimization',   bool,   True,  "with_memory_optimization.")
def train(args,
          data_args,
          train_file_list,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir,
          pretrained_model=None,
          with_memory_optimization=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image = fluid.layers.data(name='image',shape=[3,1920,1080], dtype='float32')
    size=[1920,1080]
    ground_truth = fluid.layers.data(name='ground_truth',shape=[1,size[1],size[0]], dtype='float32')
    csr_net = net.CSRNet(image,size)
    cost = fluid.layers.cos_sim(csr_net,ground_truth)
    avg_cost = fluid.layers.mean(x=cost)
    epocs = 2859 / batch_size
    optimizer = fluid.optimizer.SGD(1e-6)
    optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
    if with_memory_optimization:
       fluid.memory_optimize(fluid.default_main_program())
    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_gpu, loss_name=avg_cost.name)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, ground_truth])

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)

    best_map = 0.
    train_num = 0
    total_train_time = 0.0
    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        # end_time = 0
        every_pass_loss = []
        iter = 0
        pass_duration = 0.0
        for batch_id, data in enumerate(train_reader()):
            size=data[0][0].size
            prev_start_time = start_time
            start_time = time.time()
            if args.for_model_ce and iter == args.iterations:
                break
            if len(data) < (devices_num * 2):
                print("There are too few data to train on all devices.")
                continue
            if args.parallel:
                loss_v, = train_exe.run(fetch_list=[avg_cost.name],
                                        feed=feeder.feed(data))
            else:
                loss_v, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])
            # end_time = time.time()
            loss_v = np.mean(np.array(loss_v))
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v, start_time - prev_start_time))

            if args.for_model_ce and iter >= args.skip_batch_num or pass_id != 0:
                batch_duration = time.time() - start_time
                pass_duration += batch_duration
                train_num += len(data)
                every_pass_loss.append(loss_v)
                iter += 1
        total_train_time += pass_duration

        if args.for_model_ce and pass_id == num_passes - 1:
            examples_per_sec = train_num / total_train_time
            cost = np.mean(every_pass_loss)
            with open("train_speed_factor.txt", 'w') as f:
                f.write('{:f}\n'.format(examples_per_sec))
            with open("train_cost_factor.txt", 'a+') as f:
                f.write('{:f}\n'.format(cost))

        #best_map = test(pass_id, best_map)
        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id))
    #print("Best test map {0}".format(best_map))


if __name__ == '__main__':
    args = parser.parse_args()
    train_file_list='annotation/annotation_train_stage2.json'
    data_dir = args.data_dir
    model_save_dir = args.model_save_dir
    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=args.apply_distort,
        apply_expand=args.apply_expand,
        ap_version = args.ap_version)
    train(
        args,
        data_args=data_args,
        train_file_list=train_file_list,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=model_save_dir,
        pretrained_model=args.pretrained_model,
        with_memory_optimization=args.with_memory_optimization)
