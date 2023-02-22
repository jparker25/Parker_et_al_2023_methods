import numpy as np

def poisson_spike_train(rate, t_stop, t_start=0):
    """
    Generates a Poisson spike train with a given rate and duration.
    :param rate: The rate of the spike train (in Hz).
    :param t_stop: The duration of the spike train (in seconds).
    :param t_start: The start time of the spike train (in seconds).
    :return: A list of spike times (in seconds).
    """
    # Generate inter-spike intervals
    isi = np.random.exponential(1.0 / rate, int(rate * t_stop))
    # Generate spike times
    spikes = np.cumsum(isi) + t_start
    # Keep spikes within the desired duration
    spikes = spikes[spikes < t_stop]
    return spikes