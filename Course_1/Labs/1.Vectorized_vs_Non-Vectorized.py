# Non-vectorized vs Vectorized calculation:
import numpy as np
import time

arr1 = np.random.rand(1000000)
arr2 = np.random.rand(1000000)

c = 0
tic = time.time()
for i in range(1000000):
    c += arr1[i] * arr2[i]

toc = time.time()

print("For Loop:" + str(1000 * (toc-tic)) + "ms")

tic = time.time()
c = np.dot(arr1, arr2)
toc = time.time()

print("Vectorized version:" + str(1000 * (toc-tic)) + "ms")
