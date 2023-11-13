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


if __name__ == '__main__':
  # Dummy data
  dummy_data = torch.randn((3, 256, 256))  # Assuming image data


  # Function to profile
  def profiled_transform():
    for _ in range(1000):  # Adjust the number of iterations as needed
      transformed_data = your_transform(dummy_data)


  # Run the profiler
  cProfile.run("profiled_transform()", sort="cumulative")