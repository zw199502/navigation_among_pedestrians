import numpy as np

file_name = './results/step_980000.npy'
success_rate = np.load(file_name)
print(success_rate)