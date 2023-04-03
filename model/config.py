import tensorflow as tf

# Training/display settings
RUNTIME_SETTINGS = {
    'data_path': ".",
    'save_interval': int(1e6),
    'n_iter': int(1e20),
    'n_threads': 16,
}

# Hyperparameters
HYPERPARAMS = {
    'thedtype': tf.float32,    
    'max_nwords': 100,
    'embedding_size': int(100),
    'init_mean': 48.116,
    'init_data_variance': 573.9727,
    'init_prior_variance': 2.86002,
    'learning_rate': 258.41458,
    'nbatch': int(25.5558),
}
