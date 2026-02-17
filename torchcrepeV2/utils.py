from __future__ import division
from __future__ import print_function

import numpy as np
import torch

CENTS_MAPPING = torch.linspace(0, 7180, 360) + 1997.3794084376191


def to_local_average_cents(salience, center=None, fmin=None, fmax=None):
    """
    find the weighted average cents near the argmax bin
    """
    if fmin is not None or fmax is not None:
        pass
    probs = torch.nn.functional.sigmoid(salience)
    indices = salience.argmax(dim=-1)
    start = torch.clamp(indices - 4, min=0).unsqueeze(1)
    end = torch.clamp(indices + 5, max=salience.shape[-1]).unsqueeze(1)
    muster = torch.arange(salience.shape[1], device=salience.device).unsqueeze(0)
    mask = torch.where(muster < end, start <= muster, 0)
    
    weights_sum = (mask * probs).sum(dim=-1)
    product_sum = (CENTS_MAPPING.to(device=salience.device) * mask * probs).sum(dim=-1)
    
    return product_sum / weights_sum


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.CategoricalHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])