from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

def step_decay_schedule(initial_lr=0.001, decay_factor=0.5, step_size=10):
    """Step decay learning rate schedule"""
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return lr
    return schedule

def cosine_annealing_schedule(initial_lr=0.001, epochs=50):
    """Cosine annealing learning rate schedule"""
    def schedule(epoch):
        cosine = np.cos(np.pi * epoch / epochs)
        lr = initial_lr * (1 + cosine) / 2
        return lr
    return schedule

def get_advanced_scheduler():
    """Get advanced learning rate scheduler"""
    lr_schedule = LearningRateScheduler(
        step_decay_schedule(initial_lr=0.001, decay_factor=0.5, step_size=10)
    )
    return lr_schedule