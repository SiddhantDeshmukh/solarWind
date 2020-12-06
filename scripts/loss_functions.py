import numpy as np


def mse(array1: np.array, array2: np.array) -> float:
  # Calculate mean squared error between 'array1' and 'array2'
  # Assert that shapes of arrays are equivalent
  assert array1.shape == array2.shape, \
    f"{array1.shape} does not equal {array2.shape}"
  error = (1 / len(array1)) * np.sum((array1 - array2)**2)
  
  return error


def rmse(array1: np.array, array2: np.array) -> float:
  # Calculate root mean squared error between 'array1' and 'array2'
  # Assert that shapes of arrays are equivalent
  assert array1.shape == array2.shape,\
    f"{array1.shape} does not equal {array2.shape}"
  error = np.sqrt(mse(array1, array2))
  
  return error
