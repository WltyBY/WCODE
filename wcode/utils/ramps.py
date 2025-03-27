import numpy as np


class ramps(object):
    def __init__(
        self,
        start_iter: int,
        end_iter: int,
        start_value: float,
        end_value: float,
        mode: str,
    ):
        """
        inputs:
            start_iter: The start iteration.
            end_iter: The end itertation.
            start_value: The start values of ramps.
            end_value: The end values of ramps.
            mode: Valid values are {`linear`, `sigmoid`, `cosine`}.
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_value = start_value
        self.end_value = end_value
        self.mode = mode
        assert 0 <= self.start_iter < self.end_iter
        if self.start_value < self.end_iter:
            self.ramp_lst = self.get_rampup_ratio()
        elif self.start_value > self.end_iter:
            self.ramp_lst = self.get_rampdown_ratio()
        else:
            raise ValueError("Why self.start_value is equal to self.end_value?")

    def get_value(self, step):
        step = np.clip(step, self.start_iter, self.end_iter)
        return self.ramp_lst[step - self.start_iter]

    def get_rampup_ratio(self):
        i = np.arange(self.start_iter, self.end_iter + 1)
        if self.mode == "linear":
            rampup = (i - self.start_iter) / (self.end_iter - self.start_iter)
        elif self.mode == "sigmoid":
            phase = 1.0 - (i - self.start_iter) / (self.end_iter - self.start_iter)
            rampup = np.exp(-5.0 * phase * phase)
        elif self.mode == "cosine":
            phase = 1.0 - (i - self.start_iter) / (self.end_iter - self.start_iter)
            rampup = 0.5 * (np.cos(np.pi * phase) + 1)
        else:
            raise ValueError("Undefined rampup mode {0:}".format(self.mode))
        return rampup * (self.end_value - self.start_value) + self.start_value

    def get_rampdown_ratio(self):
        i = np.arange(self.start_iter, self.end_iter + 1)
        if self.mode == "linear":
            rampdown = 1.0 - (i - self.start_iter) / (self.end_iter - self.start_iter)
        elif self.mode == "sigmoid":
            phase = (i - self.start_iter) / (self.end_iter - self.start_iter)
            rampdown = np.exp(-5.0 * phase * phase)
        elif self.mode == "cosine":
            phase = (i - self.start_iter) / (self.end_iter - self.start_iter)
            rampdown = 0.5 * (np.cos(np.pi * phase) + 1)
        else:
            raise ValueError("Undefined rampdown mode {0:}".format(self.mode))
        return rampdown * (self.start_value - self.end_value) + self.end_value


if __name__ == "__main__":
    ramp = ramps(10, 19, 0, 2, mode="sigmoid")
    values = []
    for i in range(30):
        values.append(ramp.get_value(i))
    print(values)