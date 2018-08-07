import tensorflow as tf
from sklearn.datasets import make_moons
import numpy as np


def reset_graph(seed=13):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


X, y = make_moons(n_samples=100, noise=0.15, random_state=13)
