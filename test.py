import numpy as np
import impyute.imputation.cs.mice as mice

n = 5
arr = np.random.uniform(high=6, size=(n, n))
for _ in range(3):
    arr[np.random.randint(n), np.random.randint(n)] = np.nan
for i in range(5):
    arr[i][1] = np.nan
print(arr)
print("MICE:")
print(mice(arr))