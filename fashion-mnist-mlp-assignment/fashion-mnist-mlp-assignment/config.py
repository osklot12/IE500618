# setting static variables for easier configuration

# VARIOUS
RND_SEED = 42

# MODEL
INPUT_SHAPE = (28, 28)
N_OUTPUTS = 10

N_HIDDEN_MIN = 0
N_HIDDEN_MAX = 30
N_HIDDEN_DEFAULT = 2

N_NEURONS_MIN = 8
N_NEURONS_MAX = 256

OPTIMIZER_CHOICES = ["adam", "sgd"]

LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 1e-2

ACTIVATION_CHOICES = ["swish", "gelu"]

INITIALIZER_CHOICES = ["he_normal"]

LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 10000

# TUNER
MAX_EPOCHS = 10

FACTOR = 3

HYPERBAND_ITERATIONS = 2

# SEARCH
EARLY_STOPPING_PATIENCE = 2

EPOCHS = 10

ROOT_LOG_DIR = "../logs"

PROJECT_NAME = "hyperband"