import json
import numpy as np

file = './anonyed_every_day_traces.json'

with open(file, 'r') as f:
    data = json.load(f)

print(data[:10])
print(len(data))

log_delta_t = []
for item in data:
    for i in range(len(item)-1):
        delta_ = (item[i+1][1] - item[i][1])/1000.0  
        log_delta_t.append(np.log(delta_ + 1))
        
print(len(log_delta_t), min(log_delta_t), max(log_delta_t))
mea = np.mean(np.array(log_delta_t))
std = np.std(np.array(log_delta_t))

print(mea, std)
# day_trip_len = []

# for day_traces in data:
#     if len(day_traces) == 1335:
#         print(day_traces)
#     day_trip_len.append(len(day_traces))

# print(f'最大出行点数: {max(day_trip_len)}')

# avg_len = sum(day_trip_len) / len(day_trip_len)

# print(f'平均出行点数: {avg_len}')

# print(day_trip_len[:10])

# arr = np.array(day_trip_len)

# uniques, counts = np.unique(arr, return_counts=True)

# print(uniques)
# print(counts)

# print(sum(counts))

# total = sum(counts)

# culmulative = 0
# for u, c in zip(uniques, counts):
#     culmulative += c
#     print(f'截止出行点数 {u} 的cdf: {culmulative / total:.4f}')