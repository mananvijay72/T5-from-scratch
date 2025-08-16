import cupy as cp
import time

start = time.time()
a = cp.random.rand(10000, 10000)
b = cp.random.rand(10000, 10000)
c = cp.matmul(a, b)
cp.cuda.Device(0).synchronize()
end = time.time()

print("GPU matrix multiplication completed in", round(end - start, 2), "seconds")