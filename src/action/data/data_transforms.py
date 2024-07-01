import numpy as np


class Transform(object):
  """Abstract base class for transforms."""

  def __call__(self, *args):
    raise NotImplementedError

  def __repr__(self):
    raise NotImplementedError


class ZScore(Transform):
    """z-score channel activity."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : np.ndarray
            input shape is (time, n_channels)

        Returns
        -------
        np.ndarray
            output shape is (time, n_channels)

        """
        sample -= np.mean(sample, axis=0)
        std = np.std(sample, axis=0)
        sample[:, std > 0] = (sample[:, std > 0] / std[std > 0])
        return sample

    def __repr__(self):
        return 'ZScore()'

class NormalizeAlongDimension(Transform):
    def __init__(self, dim=-1):
        """
        Initializes the normalization transform.

        Args:
        - dim (int): The dimension along which to normalize. Default is -1 (last dimension).
        """
        self.dim = dim

    def __call__(self, x):
        """
        Applies the normalization to the input tensor.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The normalized tensor.
        """
        # Compute the sum along the specified dimension, keep dimensions for broadcasting
        sum_along_dim = x.sum(self.dim, keepdims=True)
        
        # Normalize by dividing the tensor by the sum along the specified dimension
        normalized_x = x / sum_along_dim
        
        return normalized_x
if __name__ == '__main__':
  # Dummy data
  dummy_data = torch.randn((3, 256, 256))  # Assuming image data


  # Function to profile
  def profiled_transform():
    for _ in range(1000):  # Adjust the number of iterations as needed
      transformed_data = your_transform(dummy_data)


  # Run the profiler
  cProfile.run("profiled_transform()", sort="cumulative")