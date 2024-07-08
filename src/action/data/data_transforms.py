import numpy as np


class Transform(object):
  """Abstract base class for transforms."""

  def __call__(self, *args):
    raise NotImplementedError

  def __repr__(self):
    raise NotImplementedError


class ZScore(Transform):
    """z-score channel activity."""

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, x):
        """

        Parameters
        ----------
        x : np.ndarray
            input shape is (time, n_channels)

        Returns
        -------
        np.ndarray
            output shape is (time, n_channels)

        """
        mean = np.mean(x, axis=self.dim, keepdims=True)

        # Subtract the mean from the array
        sample = x - mean

        # Calculate the standard deviation along the specified dimension
        std = np.std(sample, axis=self.dim, keepdims=True)

        # Normalize only where std is greater than 0
        nonzero_std_mask = std > 0
        sample = np.divide(sample, std, where=nonzero_std_mask, out=sample)

        return sample

    def __repr__(self):
        return 'ZScore()'

class SumAlongDimension(Transform):
    def __init__(self, dim=-1):
        """
        Sums along dimension dim
        Args:
        - dim (int): The dimension along which to sum. Default is -1 (last dimension).
        """
        self.dim = dim

    def __call__(self, x):
        """
        Applies the normalization to the input tensor.

        Args:
        - x (torch.Tensor): The input tensor.
            input shape is (time, n_days, n_channels)

        Returns:
        - torch.Tensor: The normalized tensor.
        """
        # Compute the sum along the specified dimension, keep dimensions for broadcasting
        sample = x.sum(self.dim)
        return sample

    def __repr__(self):
        return f'SumAlongDimension(%d)' % self.dim

class AvgAlongDimension(Transform):
    def __init__(self, dim=-1):
      """
      Average along dimension dim
      Args:
      - dim (int): The dimension along which to average
      """
      self.dim = dim

    def __call__(self, x):
      """
      Averages the input tensor along dim

      Args:
      - x (torch.Tensor): The input tensor.
          input shape is (time, n_days, n_channels)

      Returns:
      - torch.Tensor: The normalized tensor.
      """
      # Normalize along dimension:
      x /= x.sum(self.dim, keepdims=True)
      return x

    def __repr__(self):
        return f'AvgAlongDimension(%d)' % self.dim


class MinMaxDimension(Transform):
  def __init__(self):
    """
    Make data range 0-1.
    """

  def __call__(self, x):
    """
    Applies the normalization to the input tensor.

    Args:
    - x (torch.Tensor): The input tensor.
        input shape is (time, n_days, n_channels)

    Returns:
    - torch.Tensor: The normalized tensor.
    """
    # Compute the sum along the specified dimension, keep dimensions for broadcasting
    min_val = 0  # x.min()
    max_val = x.max()
    # Avoid division by zero
    range_val = max_val - min_val
    assert range_val > 0

    # Apply min-max normalization
    normalized_sample = (x - min_val) / range_val

    return normalized_sample

  def __repr__(self):
    return f'MinMaxDimension(%d)' % self.dim


if __name__ == '__main__':
  # Dummy data
  dummy_data = torch.randn((3, 256, 256))  # Assuming image data


  # Function to profile
  def profiled_transform():
    for _ in range(1000):  # Adjust the number of iterations as needed
      transformed_data = your_transform(dummy_data)


  # Run the profiler
  cProfile.run("profiled_transform()", sort="cumulative")