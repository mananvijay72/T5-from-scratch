# tests/test_tensor.py
from core.tensor import Tensor
from core.ops import add, matmul, relu
import numpy as np

def test_add():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = add(a, b)
    c.backward(np.array([1.0, 1.0]))
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])

def test_matmul():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0], [4.0]], requires_grad=True)
    c = matmul(a, b)
    c.backward(np.array([[1.0]]))
    assert np.allclose(a.grad, [[3.0, 4.0]])
    assert np.allclose(b.grad, [[1.0], [2.0]])

def test_relu():
    a = Tensor([-1.0, 2.0], requires_grad=True)
    b = relu(a)
    b.backward(np.array([1.0, 1.0]))
    assert np.allclose(a.grad, [0.0, 1.0])