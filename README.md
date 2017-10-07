Requirements:
=============

 * Python 3.6+
 * TensorFlow 1+
 * Numpy
 * Pickle

Saving model
============

Save model into model_data/prod. Only 1 checkpoint is needed to function, but make sure to include "checkpoint" file is included, and proper checkpoint is listed. All files with the same index need to be saves in prod.

dev folder is for training, and nothing is committed in git (see .gitignore)

Misc notes
----------

Original image: 1280x720

Symmetrical sizes: 640x360, 320x180, 160x90 < this corresponds to /8 reduction, maximum for TensorFlow

When re-training the model, need to make sure to delete all training history first, as saved checkpoints will interfere.
And if network architecture is changed, it will not work at all, as variables will be different than in checkpoint.
