---
title: backward 操作时 grad_tensors 参数的作用
author: me
date: 2024-10-25 20:45:00 +0800
categories: [Deep Learning, PyTorch]
tags: [Deep Learning, PyTorch, Gradient]
math: true
---

## 问题提出

在阅读 Dive Into Deep Learning 时，我在 [ReLU激活函数部分](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp.html#relu) 看到这样一段代码：

```python
import torch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
y.backward(torch.ones_like(x), retain_graph=True)
```

其中在对非标量张量 y 进行梯度计算时，引入了参数 `(grad_tensors=)torch.ones_like(x)`。如果不加入这个参数，在运行时则会报错：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  
y = x*x 
y.backward()
```

```text
RuntimeError: grad can be implicitly created only for scalar outputs
```

证明在计算非标量张量对张量的梯度时，必须引入 `grad_tensors` 这一参数。但是为什么要引入这一参数？这个参数是如何参与到梯度计算中的？

阅读 [torch.autograd.backward 文档](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)，发现 PyTorch 这样解释计算非标量张量对张量的梯度的原理：

>The graph is differentiated using the chain rule. If any of `tensors` are non-scalar (i.e. their data has more than one element) and require gradient, then the Jacobian-vector product would be computed, in this case the function additionally requires specifying `grad_tensors`. It should be a sequence of matching length, that contains the “vector” in the Jacobian-vector product, usually the gradient of the differentiated function w.r.t. corresponding tensors (`None` is an acceptable value for all tensors that don’t need gradient tensors).  
>
>该（计算）图使用链式法则进行微分。如果任何`tensors`是非标量（即它们的数据具有多个元素）并且需要梯度，则将计算雅可比向量积，在这种情况下，该函数还需要指定`grad_tensors` 。它应该是一个匹配长度的序列，包含雅可比向量乘积中的“向量”，通常是对应张量的微分函数的梯度（对于所有不需要梯度张量的张量来说， `None`是可接受的值）。

## 背景知识：雅可比矩阵

在一般的梯度计算中，如果要计算矩阵对矩阵的梯度，会借助雅各比矩阵来进行。例如，假设我们有一个函数 $$f(x)$$，其中：
- 输入张量 $$x$$ 的形状为 $$[p]$$。
- 输出张量 $$y = f(x)$$ 的形状为 $$[n]$$。

那么，$$y$$ 对 $$x$$ 的雅可比矩阵 $$J$$ 的维度是 $$[n, p]$$。这表示 $$y$$ 中的每个元素 $$y_{i}$$ 对 $$x$$ 的偏导数 $$\frac{\partial y_{i}}{\partial x_k}$$ 构成了雅可比矩阵的元素 $$J_{i,k}$$。在这个雅可比矩阵中，张量 $$x$$ 的每个元素都对应了 $$n$$ 个梯度值。

$$
J = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_p} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_p} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_n}{\partial x_1} & \frac{\partial y_n}{\partial x_2} & \cdots & \frac{\partial y_n}{\partial x_p}
\end{bmatrix}
$$

但这样做存在一些问题：

1. 没有定义如何将每个 $$x_i$$ 的若干梯度值计算为单个梯度
2. 计算这么大的雅可比矩阵会消耗较大的内存和计算量。

所以在计算非标量张量对非标量张量的梯度时，PyTorch 不会直接计算雅可比矩阵（Jacobian Matrix），而是通过 `grad_tensors` 实现了对雅可比向量积的高效计算。这不仅节省内存和计算量，而且满足大多数深度学习应用的需求。

## 背景知识：雅可比向量积

**雅可比向量积**（JVP）则是将雅可比矩阵 $$J$$ 与一个向量 $$v$$ 相乘，即：

$$
J \cdot v
$$

其中 $$v$$ 的维度与 $$y$$ 相同。这一操作的结果是一个与 $$x$$ 相同维度的向量，即输出张量的加权导数之和。通过计算雅可比向量积，而不是完整的雅可比矩阵，我们可以避免计算和存储所有的偏导数。

$$
J \cdot v = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_1} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_p} & \cdots & \frac{\partial y_n}{\partial x_p}
\end{pmatrix}
\begin{pmatrix}
\frac{\partial l}{\partial y_1} \\
\vdots \\
\frac{\partial l}{\partial y_n}
\end{pmatrix}
= \begin{pmatrix}
\frac{\partial l}{\partial x_1} \\
\vdots \\
\frac{\partial l}{\partial x_p}
\end{pmatrix}
$$

## grad_tensors 和雅可比向量积的关系

在 PyTorch 中，`grad_tensors` 参数实际上就是充当这个向量 $$v$$ 的角色。在计算梯度时，如果我们传入了 `grad_tensors`，则 PyTorch 计算的梯度实际上是雅可比矩阵 $$J$$ 与 `grad_tensors` 的向量积 $$J \cdot \text{grad\_tensors}$$，也就是：

$$
\frac{\partial (y \cdot \text{grad\_tensors}^{\mathrm T})}{\partial x}
$$

这种方式高效地计算了雅可比向量积，而不是直接计算整个雅可比矩阵，从而避免了显式存储和计算高维的雅可比矩阵。

## 总结

通过使用 `grad_tensors`，PyTorch 可以高效地计算非标量张量对输入的梯度。其核心思想是利用雅可比向量积来避免显式构造雅可比矩阵，从而在内存和计算上实现了极大的优化。这种方式在深度学习中尤为有用，因为直接构造和存储高维雅可比矩阵不仅昂贵且不切实际。

## References

1. [torch.autograd.backward - PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
2. [Pytorch autograd,backward详解](https://zhuanlan.zhihu.com/p/83172023)
3. [我的 ChatGPT 提问过程](https://chatgpt.com/share/671b5b97-bd64-8007-a484-4394ed518190)
