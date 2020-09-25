class Params:
    def __init__(self, eps, alpha, ucb, sample_average, initial_values, confidence_level = 2):
        self.epsilon = eps
        self.alpha = alpha
        self.ucb = ucb
        self.initial_values = initial_values
        self.sample_average = sample_average
        self.confidence_level = confidence_level