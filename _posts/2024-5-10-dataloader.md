---
title: 理解 PyTorch 中 DataLoader 与 Dataset 的使用
author: me
date: 2024-05-10 22:00:00 +0800
categories: [Machine Learning, PyTorch]
tags: [machine learning, deep learning, data processing, python, pytorch, dataloader, dataset]
---

为了提高代码的可读性与模块化特性，我们希望数据集代码与模型训练代码分离。于是 PyTorch 提供了两个原始类型（Data Primitive）：`torch.utils.data.DataLoader` 与 `torch.utils.data.Dataset`，分别用于定义数据集对象、迭代读取数据条目。

下面将先介绍如何快速上手，之后对两个原始类型的参数作详细解释。

## 快速上手

### 实现 Dataset 子类

> `Dataset` 是抽象基类，需要以它为基类编写子类，接着将子类实例化。
{: .prompt-tip}

这部分代码需要根据实际的数据形式进行调整。下面只是一个简单的示例：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, source_arr, target_arr):
        super().__init__()

        num_sample = len(source_arr)

        self.source_arr = source_arr
        self.target_arr = target_arr
        self.num_sample = num_sample

    def __getitem(self, idx):
        return self.source_arr[idx], self.source_arr[idx]
    
    def __len__(self):
        return self.num_sample
```
{: file="dataloader/dataset.py" }

### 导入类

导入刚刚写的 `Dataset` 子类与 `DataLoader`。

``` python
from dataloader.dataset import MyDataset as Dataset
from torch.utils.data import DataLoader
```
{: file="main.py" }

### 实例化 Dataset 的子类对象

在主程序中，调用已经实现的 `Dataset` 子类来将其实例化。

```python
from dataloader import MyDataset as Dataset

train_dataset = Dataset(source_arr, target_arr)
```
{: file="main.py" }

### 实例化 DataLoader 对象

在主程序中调用 `DataLoader` 类进行实例化，并传入 `Dataset` 对象。

```python
train_dataloader = DataLoader(train_dataset, arg1, arg2, ...)
```
{: file="main.py" }

与其在创建对象的时候填写参数，更推荐的是提前将参数打包成字典，在创建对象时进行解包：

```python
dataloader_args = {"batch_size": 256, "shuffle": True, "num_workers": 8}
train_dataloader = DataLoader(train_dataset, **dataloader_args)
```
{: file="main.py" }

### 迭代 DataLoader 对象

```python
for sample in train_dataloader:
    ...
```
{: file="main.py" }

`DataLoader` 的迭代返回值取决于 `collate_fn` 参数。

如果 `collate_fn == None`，则对原始 batch 值使用默认函数进行处理并返回。否则，将原始 batch 值使用 `collate_fn` 指定的函数进行处理并返回。

具体来看，对于下面的最小测试单元：

```python
from dataloader.dataset import MyDataset as Dataset
from torch.utils.data import DataLoader

source_arr = [1, 2, 3, 4, 5]
target_arr = [1, 2, 3, 4, 5]

dataset = Dataset(source_arr, target_arr)
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

for i, (source, target) in enumerate(dataloader):
    print(f'Batch {i + 1}:')
    print(f'Source: {source}')
    print(f'Target: {target}')
    print()
```
{: file="test.py" }

处理得到的原始 batch 值形如：

```python
[(1, 1), (2, 2), (3, 3)]
```
{: file="terminal" }

因为没有指定 `collate_fn`，控制台的输出内容是默认函数处理的结果。

```
Batch 1:
Source: tensor([1, 2, 3])
Target: tensor([1, 2, 3])

Batch 2:
Source: tensor([4, 5])
Target: tensor([4, 5])
```
{: file="terminal" }

#### 创建 collate_fn 函数

如果希望对数据进行进一步处理，可以创建一个 `collate_fn()` 函数，这样迭代返回值就是原始 batch 值经过这个函数处理后的内容。

> 一般将这个函数写在 `MyDataset` 类里，并使用 `@staticmethod` 装饰器。
{: .prompt-tip}

```python
class MyDataset(Dataset):
    ...
    
    @staticmethod
    def collate_fn(batch):
        # Your code here
        return ...
```
{: file="dataloader/dataset.py" }

如果希望在未指定 `collate_fn` 时的默认结果基础上处理，可以通过下面的函数实现。

```python
    @staticmethod
    def collate_fn(batch):
        source, target = zip(*batch)
        source = np.stack(source)
        target = np.stack(target)

        # Your code here

        return source, target
```
{: file="dataloader/dataset.py" }

## 介绍

### Dataset

`Dataset` 的作用是传入 `DataLoader` 供其包装、迭代。但它是一个抽象类（Abstract Class），意味着需要以 `Dataset` 为基类创建一个新的子类。

> PyTorch 要求子类必须重写 `__getitem__()` 方法。
> 
> 可选实现 `__len__()` 和 `__getitems__()` 方法，来提高性能表现。
{: .prompt-info }

### DataLoader

`DataLoader` 接收 `Dataset` 的子类对象作为数据集，自身为可迭代对象。其提供了较多可选参数，下面进行介绍。

1. **dataset** (*Dataset*): 加载数据的数据集。
2. **batch_size** (*int, optional*): 每个 batch 有多少个样本。默认为 `1`。
3. **shuffle** (*bool, optional*): 在每个 epoch 对数据重新排序。默认为 `False`。
4. **sampler** (*Sampler or Iterable, optional*): 自定义抽取样本的策略。与 `shuffle` 互斥。
5. **batch_sampler** (*Sampler or Iterable, optional*): 类似于 `sampler`，但一次返回一批索引。与 `batch_size`, `shuffle`, `sampler`, `drop_last`等参数互斥。
6. **num_workers** (*int, optional*): 用于数据加载的子进程（worker）数。设置为 `0` 表示数据将在主进程中加载，大于 `0` 表示使用多个子进程加载数据。默认为 `0`。
7. **collate_fn** (*Callable, optional*): 这个函数合并一个 `list` 类型样本来形成一个 `mini-batch`。
8. **pin_memory** (*bool, optional*): 如果为 `True`，将在返回 Tensors 之前将其复制到设备/CUDA 固定内存中。
9. **drop_last** (*bool, optional*): 设置为 `True` 会丢弃最后一个不完整的批次。默认为 `False`。
10. **timeout** (*numeric, optional*): 从工作进程收集批次的超时值。默认为 `0`。
11. **worker_init_fn** (*Callable, optional*): 每个 worker 初始化的函数，一般不需要自己设置。默认为 `None`。
12. **multiprocessing_context** (*str or multiprocessing.context.BaseContext, optional*): 多进程上下文，一般使用操作系统的默认上下文即可。
13. **generator** (*torch.Generator, optional*): 用于生成随机索引的随机数生成器，一般不需要自己设置。
14. **prefetch_factor** (*int, optional, keyword-only arg*): 每个工作进程提前加载的批次数，可以提高数据加载效率。如果 `num_workers=0` 默认值为 `None`。否则，如果 `num_workers > 0` 默认值为 `2`。
15. **persistent_workers** (*bool, optional*): 如果为 `True`，不会在数据集被使用一次后关闭工作进程。这允许保持工作数据集实例处于活动状态。默认为 `False`。
16. **pin_memory_device** (*str, optional*): 如果 `pin_memory` 为 `True`，指定将数据固定到的设备。

具体了解 `DataLoader` 的数据处理流程，可以参考这两篇文章：

[阿里云：PyTorch 小课堂开课啦！带你解析数据处理全流程（一）](https://developer.aliyun.com/article/914214)

[阿里云：PyTorch 小课堂开课啦！带你解析数据处理全流程（二）](https://developer.aliyun.com/article/914199)

## References

[1] PyTorch - torch.utils.data. [Link](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

[2] PyTorch - Datasets & DataLoaders. [Link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
