# 高性能模型  
  
本篇文档和伴随的 [脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) 详细记述了怎样构建以多种系统类型和网络拓扑为目标的高度伸缩性模型。本文档中的技术使用了一些底层 TensorFlow Python 原型。将来，高级 API (应用程序接口) 会包含这些中的许多技术。  
  
## 输入管道 
  
这篇指南 (@{$performance_guide$Performance Guide}) 解释了怎样识别可能的输入管道问题以及最好的实践方法。我们发现当很大的输入和每秒高采样处理时，比如用 [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 来训练 ImageNet, @{tf.FIFOQueue} 和 @{tf.train.queue_runner} 不能充分饱和的使用多个现代 GPU。原因是由于使用了底层的 Python 线程实现，而 Python 线程开销太大。  
  
另外一个方法， 我们已实现在这个 [脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) 里，它是通过使用 TensorFlow 原生的并行机制来构建一个输入管道。我们的实现由 3 个阶段构成：  
  
* I/O 读取： 从磁盘中选择并读取图像文件。  
* 图像处理： 解码图像记录到图像，预处理后组织到小批量中。  
* CPU 到 GPU 的数据传输： 从 CPU 传输图像 到 GPU 中。  
  
每个阶段的主要部分是用 `data_flow_ops.StagingArea` 与其他阶段并行执行的。`StagingArea` 类似队列运算符，与 @{tf.FIFOQueue} 相似。不同之处在于 `StagingArea` 不是 FIFO (先进先出) 排序，而是提供了更简单功能型，并且能在 CPU 和 GPU 与其他阶段并行执行。把输入管道分解3个阶段独立并行的操作具有伸缩性，并且充分利用了大型多核环境优势。余下的章节详细记载了这些阶段和如何使用 `data_flow_ops.StagingArea` 的细节。  

### 并行化 I/O  
  
`data_flow_ops.RecordInput` 用来从磁盘并行化读取数据。给定一系列表示 TensorFlow 记录的输入文件，`RecordInput` 使用背景线程持续的的读取记录。这些记录被放入自己内部大存储池里，当加载到存储池至少一半容量后，就会产生输出张量。  
  
这些操作有自己的内部线程，这些线程由消耗最小 CPU 的 I/O 时间主导，允许和其他模型并行流畅的运行。  
  
### 并行化图像处理  
  
从 `RecordInput` 读出的图像，作为张量被传递到图像处理管道。为了使图像处理管道更容易解释，假定输入管道目标是 8 个 GPU 处理 256 记录 的批量大小（每个 GPU 处理 32 个记录）。  
  
并行化的一个个读取和处理256 个记录。 从 TensorFlow 图中 256 个独立的 `RecordInput` 读操作开始。每次读操作后面跟随一组相同的，作为并行独立执行的图像预处理操作。图形处理操作包括图像解码，变形, 缩放。  
  
一旦图像进入到预处理阶段，它们会被连接成 8 个张量，每个含 32 个批处理。如果用 @{tf.concat} 来完成这个任务，它作为一个单一操作运算，必须等待所有输入数据都准备好才开始连接。但是如果用 @{tf.parallel_stack} 的话，它会分配一个未初始化的输出张量，每个输入张量在准备好后会立刻写入到输出张量的相应位置。
  
当所有输入张量处理完成后，输出张量被传递进图中。这有效的隐藏了产生所有输入张量的长尾性的所有内存延迟。  
  
### 并行化 CPU 到 GPU 数据传输  
  
继续假定目标是处理 256 批次大小的 8 个 GPU（每个 GPU 处理 32 个）。一旦输入图像被 CPU 处理和连接起来，我们就得到了 8 个批次大小为 32 的张量。  
  
TensorFlow 允许一个设备中的张量能直接在另外一个设备中使用。当需要时，TensorFlow 通过插入隐式拷贝使得张量能用在任何设备中。张量在真正使用之前，运行时在设备间调度拷贝。然而，如果拷贝不能及时完成，需要这些张量的计算就会停滞，导致性能的降低。  
  
运行时，`data_flow_ops.StagingArea` 用来显式的并行调度拷贝。最终结果是当所有 GPU 计算开始后，所有的张量可以被使用。  
  
### 软件管道  
  
伴随着各种处理器驱动所有的操作，`data_flow_ops.StagingArea` 能在所有处理中使用，因此能并行化的运行。`StagingArea` 跟 @{tf.FIFOQueue} 相似，是个类队列的操作符，能在 CPU 和 GPU 之间执行简单功能。  
  
模型运行所有阶段之前，输入管道能在准备不同的数据预备阶段缓存一组数据。每个步骤执行时，开始时从缓存区读取一批数据，结束时再将它们放入缓存区。  
  
例如： 假设有 3 个阶段： A, B, C。如果中间有两个阶段区域： S1 和 S2。在准备动作期间，我们运行：  
  
'''  
准备动作：  
步骤 1: A0  
步骤 2: A1 B0  
  
实际执行：   
步骤 3: A2  B1  C0  
步骤 4: A3  B2  C1  
步骤 5: A4  B3  C2  
'''  
  
准备开始阶段后，S1 和 S2 其中每个有一套数据。实际执行的每个步骤，一套数据从每个阶段区域中被执行完，另一套数据就会添加进去。  
  
使用这种方案的好处是：  
  
* 所有的阶段都是非阻塞的，因为准备动作开始后阶段区域总是有一套数据等待被处理。  
* 每个阶段能立即开始，所以能并行执行。  
* 阶段缓存总有个固定内存开销。它们最多会有一套额外数据。  
* 仅仅单个 `session.run()` 调用被输入运行所有步骤阶段，使得数据配置和调试更容易。  
  
## 构建高性能模型的最好实践  
  
下面搜集了一些能提供性能和增强模型的灵活性的最佳实践。。  
  
### 使用 NHWC 和 NCHW 构建模型  
  
大部分 CNN (卷积神经网络）使用的 TensorFlow 操作都支持 NHWC 和 NCHW 数据格式。在 GPU 上，NCHW 更快。但是 在 CPU 上，NHWC 有时候更快。  
  
构建一个不论在什么平台上都能保持模型的灵活性和操作最优，且支持这两个数据格式的模型。大部分 CNN 使用的 TensorFlow 操作都支持 NHWC 和 NCHW 数据格式。基准测试脚本支持 NCHW and NHWC 写入。训练 GPU 时应当总是用 NCHW。NHWC 在 CPU 上有时候更快。一个灵活的模型能在 GPU 上用 NCHW 训练，在 CPU 上能用训练获得的权重使用 NHWC 做推断。  
  
### 使用融合批规范化  
  
TensorFlow 中的默认批次规范化是复合操作。虽然这很通用，但是常常导致次优性能。一个替代方法是使用融合的批规范化，在 GPU 上常常有更好的性能。下面是一个使用 @{tf.contrib.layers.batch_norm} 来实现融合批规范化的例子。  
  
```python  
bn = tf.contrib.layers.batch_norm(  
          input_layer, fused=True, data_format='NCHW'  
          scope=scope)  
```  
  
## 变量分布和梯度累积  
  
训练期间，训练变量的值被累计梯度和 delta 变量更新。在基准脚本中，我们展示了大范围性能分布和累积方案能使用灵活和通用的 TensorFlow 原型来构建。  
  
脚本中包含下面三个变量分布和累积的例子：  
  
* `parameter_server` 中每个训练模型的副本从参数服务器中读取变量，然后独立的更新。当每个模型需要这些变量时，它们通过 TensorFlow 运行时隐式拷贝。[脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) 中的例子说明了本地训练，分布式同步训练和分布式异步训练的这些方法。  
  
* `replicated` 放置每个相同训练变量到每个 GPU 中。当变量可用时，前向和后向计算就能立即开始。梯度沿着所有 GPU 被累积，累积总和应用到每个 GPU 的变量副本以保持同步。
  
* `distributed_replicated` 放置一个相同的训练参数拷贝到每个 GPU 中，并且放置一份主机拷贝到参数服务器中。当变量数据立即可用时，前向和后向计算就能立即开始。梯度沿着每个服务器上 所有 GPU 被累积， 然后每个服务器累积的梯度应用到主机拷贝中。所有的从机做完这些，每个从机从主机拷贝中更新它的变量拷贝。  
  
下面是每个方法的额外细节：  
  
### 参数服务器变量  
  
最常见的 TensorFlow 管理训练变量的方式是参数服务器模型。  
  
在一个分布式系统中，每个从机进程运行相同的模型，参数服务器进程拥有主机变量拷贝。当一个从机需要参数服务器的变量时，它就会直接指向它。TensorFlow 运行时间通过增加隐式的拷贝到图中来使得变量值在需要它的计算设备中可用。当一个从机计算了一个梯度，它就会发送给拥有这个特定变量的参数服务器，相对应的优化器就来更新这个变量。  
  
通过下面一些技术来提高：  
  
* 基于变量大小在服务器参数中传播这些变量来实现均衡加载。  
* 每个从机有多个 GPU 时，梯度可能沿着所有 GPU 累积，然后单个累积梯度发送给参数服务器。这样能减少网络带宽和参数服务器的工作量。  
  
为了协调从机，常用的模型是异步更新。每个从机更新主机变量拷贝而不需要与其它从机同步。在我们的模型中，我们展示了，很容易地引进从机间的同步，因此下一步开始前，所有从机更新能在前一步执行完。  
  
参数服务器方法也能用于本地训练。这种情况下，代替在参数服务器中传播主机变量拷贝，它们要么放在 CPU 中，要么在 GPU 中传播。  
  
由于这个安装的简单性，这个构建在社区内受到了很大的欢迎。  
  
在脚本中通过传递下面这个来使用这个模型：  
`--variable_update=parameter_server`.  
  
<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">  
  <img style="width:100%" alt="parameter_server mode in distributed training"  
   src="../images/perf_parameter_server_mode_doc.png">  
</div>  
  
### 复制的变量  
  
本设计中，每个服务器中的 GPU 都有每个变量的拷贝。这些值通过应用完全的累积梯度结果到每个 GPU 的变量拷贝中来保持在 GPU 中的同步。  
  
因为这些变量和数据在训练开始时就可用，所以训练的前向传递能立即开始。梯度在这些设备间累积，然后完全的梯度累积结果应用到每个本地拷贝中。  
  
服务器间的梯度累积可以用下面方式来实现：  
  
* 在单个设备 (CPU 或 GPU) 中使用标准的 TensorFlow 操作来累积总和，然后拷贝回所有的 GPU 中。  
* 使用 NVIDIA® NCCL (英伟达集合通信库), 在下面的 NCCL 章节有描述。  
  
这个模型能在脚本中通过传递 `--variable_update=replicated` 来使用。  
  
### 分布式训练中复制的变量  
  
变量复制方法能扩展到分布式训练。一种方法是使用类似的复制模型： 累积所有聚类的梯度，然后应用它们到每个变量的拷贝中。这可能会显示在这个脚本的未来版本中。本脚本中确实展现了不同的变化，如下描述所示：  
  
除了每个 GPU 的变量拷贝外，在本模式中，主机拷贝是存储在参数服务器中的。使用复制模式，训练能使用本地变量拷贝来立即开始。  
  
当这些权重梯度可用时，它们会发回到参数服务器中，更新所有的本地拷贝。  
  
1. 同一个从机里的所有的 GPU 梯度被累积在一起。  
2. 每个从机中累积的梯度发回到拥有它的参数服务器，其中指定的优化器来更新主机变量拷贝。  
3. 每个从机更新主机的本地变量拷贝。在这个例子模型中，我们通过交叉复制障碍来实现。交叉复制障碍等待所有从机完成变量更新，复制器释放障碍后，然后来取新的变量。一旦所有的变量拷贝完成，这标志着一个训练步骤的结束，新的步骤的开始。  
  
尽管这听起来类似于参数服务器的标准使用，但是多数情况下性能会更好。主要原因是实际上无延迟的计算和大量的早期的梯度拷贝延迟被隐藏在之后的计算层中。  
  
在脚本中通过传递下面这个来使用这个模型：  
`--variable_update=distributed_replicated`.  
  
<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">  
  <img style="width:100%" alt="distributed_replicated mode"  
   src="../images/perf_distributed_replicated_mode_doc.png">  
</div>  
  
#### NCCL  
  
为了在相同的主机中传播变量和累积不同 GPU 间的梯度，我们使用缺省的 TensorFlow 隐式拷贝机制。  
  
然而，我们可使用可选的 NCCL (@{tf.contrib.nccl}) 支持。NCCL 是一个在不同 GPU 间有效传播和累积数据的 NVIDIA® 库。它调度一个协作内核，知道怎样最好的利用这个底层硬件拓扑，这个内核使用单个 GPU 流式多处理器。  
  
在我们的实验中，我们显示尽管 NCCL 常常本身能导致更快的数据累积，但是并不一定产生更快的训练。我们假设隐式拷贝基本是自由的，原因是它能在 GPU 拷贝引擎中出现和它的延迟被主计算本身隐藏。尽管 NCCL 能更快传输数据，但是它耗费了一个流式多处理器，并且给底层的 L2 缓存增加了更多的压力。我们的结果显示对 8 个 GPU 而言，NCCL常产生更好的性能。然而，对更少 GPU 而言，隐式拷贝常常有更好的性能。  
  
### 阶段性变量  
  
我们进一步介绍了阶段性变量模式，其中我们使用阶段区域来操作变量读取和更新。和输入管道的软件管道相似，这能隐藏数据的拷贝延迟。如果计算时间比拷贝和累积更长，拷贝本身基本变得自由了。  
  
这个的负面性是所有权重读取都是来自前一个训练步骤。因此这是个不同于随机梯度下降的算法。但是它有可能能通过调整学习率和其他超参数来提高收敛速度。  
  
## 执行这个脚本  
  
本节列出了执行这个主脚本的核心命令行参数和一些基本的例子 (([tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py))。  
  
> 注意 `tf_cnn_benchmarks.py` 使用 `force_gpu_compatible` 这个配置，
> 在 TensorFlow 1.1 之后被引用。TensorFlow 1.2 发布前
> 建议从源代码来构建。
  
#### 基本命令行参数：  
  
*   **`model`**: 使用的模型, 例如 `resnet50`, `inception3`, `vgg16` 和 `alexnet`.  
*   **`num_gpus`**: 使用的 GPU 数量  
*   **`data_dir`**: 处理的数据路径。如果未设置，合成数据会被使用。如果要使用 Imagenet 数据，使用： [说明](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) 作为起始点。  
*   **`batch_size`**: 每个 GPU 批处理大小  
*   **`variable_update`**: 管理变量的方法: `parameter_server`, `replicated`, `distributed_replicated`, `independent`  
*   **`local_parameter_device`**: 使用参数服务器的设备: `cpu` 或者 `gpu`.  
  
#### 单实例事例：  
  
```bash  
# VGG16 通过运用参数优化 Google 计算引擎， 用 8 个 GPU 来训练 ImageNet 数据  
python tf_cnn_benchmarks.py --local_parameter_device=cpu --num_gpus=8 \  
--batch_size=32 --model=vgg16 --data_dir=/home/ubuntu/imagenet/train \  
--variable_update=parameter_server --nodistortions  
  
# VGG16 通过运用参数优化 NVIDIA DGX-1， 用 8 个 GPU 来训练 合成 ImageNet 数据  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=vgg16 --variable_update=replicated --use_nccl=True  
  
# VGG16 通过运用参数优化 Amazon EC2， 用 8 个 GPU 来训练 ImageNet 数据  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=vgg16 --variable_update=parameter_server  
  
# ResNet-50 通过运用参数优化 Amazon EC2， 用 8 个 GPU 来训练 ImageNet 数据  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=resnet50 --variable_update=replicated --use_nccl=False  
  
'''  
  
#### 分布式命令行参数  
  
*   **`ps_hosts`**: 用逗号分隔的用作参数服务器的一系列主机，形式为 ```<host>:port```, e.g. ```10.0.0.2:50000```。  
*   **`worker_hosts`**: 用逗号分隔的用作从机一系列机器，形式为 ```<host>:port```, e.g. ```10.0.0.2:50001```。  
*   **`task_index`**: 在`ps_hosts` 或 `worker_hosts` 启动的系列主机索引。  
*   **`job_name`**: 作业类型，例如 `ps` 或者 `worker`。  
  
#### 分布式例子  
  
下面是一个在 2 个主机： host_0 (10.0.0.1) 和 host_1 (10.0.0.2) 训练 ResNet-50 的例子。 这个例子使用合成数据。传递 `--data_dir` 参数来使用真实数据。  
  
```bash  
# 在 host_0 (10.0.0.1) 上运行下面命令：  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \  
--job_name=worker --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \  
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=0  

python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \  
--job_name=ps --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \  
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=0  
  
  
# 在 host_1 (10.0.0.2) 上运行下面命令：  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \  
--job_name=worker --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \  
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=1  
  
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \  
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \  
--job_name=ps --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \  
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=1  
  
```  
