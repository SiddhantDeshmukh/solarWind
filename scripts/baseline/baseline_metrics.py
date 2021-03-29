import numpy as np


def n_steps_ago(data: np.array, n=648) -> np.array:
  # Get the forecasts from 'n' steps ago for each sample.
  # 'data' is a 3D tensor with shape (nsamples, batch_size, n_features)
  output = data[n:, :, :]
  return output


def n_steps(data: np.array, n=24, repeated=False, use_last=True) -> np.ndarray:
  # 'n' last (first) steps; if 'repeated=True', takes the last (first) step
  # and repeats it 'n' times if 'use_last' is True (False)
  repeat_idx = -1 if use_last else 0
  if repeated:
    output = np.repeat(data[:, repeat_idx, :], n, axis=1)
  else:
    if use_last:
      output = data[:, -n:, :]
    else:
      output = data[:, :n, :]

  return output


def first_n_steps(data: np.ndarray, n=24, repeated=False) -> np.ndarray:
  # First 'n' steps; if 'repeated=True', takes the first step and repeats
  # it 'n' times
  if repeated:
    output = np.repeat(data[:, 0, :], n, axis=1)
  else:
    output = data[:, :n, :]

  return output


def last_n_steps(data: np.ndarray, n=24, repeated=False) -> np.ndarray:
  # Last 'n' steps; if 'repeated=True', takes the last step and repeats
  # it 'n' times
  if repeated:
    output = np.repeat(data[:, -1:, :], n, axis=1)
  else:
    output = data[:, -n:, :]

  return output


def mean_n_steps(data: np.ndarray, n=24, repeated=False, use_last=True) -> np.ndarray:
  # Mean of last (first) 'n' steps if 'use_last=True' (False),
  # repeated 'n' times if 'repeated'
  data_ = data[:, -n:, :] if use_last else data[:, n:, :]
  if repeated:
    output = np.repeat(np.reshape(np.mean(data_, axis=1),
                                  (len(data), 1, data.shape[-1])), n, axis=1)
  else:
    output = np.mean(data_, axis=1)

  return output


def median_n_steps(data: np.ndarray, n=24, repeated=False, use_last=True) -> np.ndarray:
  # Median of last (first) 'n' steps if 'use_last=True' (False),
  # repeated 'n' times if 'repeated'
  data_ = data[:, -n:, :] if use_last else data[:, n:, :]
  if repeated:
    output = np.repeat(np.reshape(np.median(data_, axis=1),
                                  (len(data), 1, data.shape[-1])), n, axis=1)
  else:
    output = np.median(data_, axis=1)

  return output
