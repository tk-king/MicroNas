ts_len = 705904
window_size = 64
stride = 32

num_windows = (ts_len - window_size) // stride + 1

print(num_windows)


