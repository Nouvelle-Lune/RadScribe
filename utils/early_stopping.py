from tqdm import tqdm
from colorama import init, Fore, Style


class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.001, mode="min"):

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.num_bad_epochs = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return

        if (
            self.mode == "min" and current_score < self.best_score - self.min_delta
        ) or (self.mode == "max" and current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.num_bad_epochs = 0
        else:
            tqdm.write(
                Fore.RED
                + f" ⚠️ EarlyStopping: Current score {current_score} did not improve from best score {self.best_score}.\n Current number of bad epochs: {self.num_bad_epochs} epochs left."
                + Style.BRIGHT
            )
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.early_stop = True
