import numpy as np

cpu_results = list()
for i in range(10):
    for j in range(32):
        sample = np.fromfile("output/cpu_fp32/Result_{}/class_probs.raw".format(i), dtype=np.float32, count=1000, offset=j * 1000)
        cpu_results.append(np.argmax(sample))

print(cpu_results)
print(len(cpu_results))

htp_results = list()
for i in range(10):
    for j in range(32):
        sample = np.fromfile("output/htp_int8/Result_{}/class_probs.raw".format(i), dtype=np.float32, count=1000, offset=j * 1000)
        htp_results.append(np.argmax(sample))

gpu_results = list()
for i in range(10):
    for j in range(32):
        sample = np.fromfile("output/gpu_fp32/Result_{}/class_probs.raw".format(i), dtype=np.float32, count=1000, offset=j * 1000)
        gpu_results.append(np.argmax(sample))

print(htp_results)
print(gpu_results)



print(htp_results == gpu_results)
print(gpu_results == cpu_results)

count = 0
for it, res in enumerate(htp_results):
    if res == cpu_results[it]:
        count += 1
print(count/len(htp_results))