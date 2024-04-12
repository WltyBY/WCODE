from torch import nn


class DeepSupervisionWeightedSummator(nn.Module):
    def __init__(self, loss, weight=None):
        """
        Calculate the weighted sum of losses for multiple (predicted, labeled) pairs
        l = w[0] * loss(input0[0], input1[0], ...) + w[1] * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWeightedSummator, self).__init__()
        self.weight = weight
        self.loss = loss

    def forward(self, *args):
        # args should be the predictions list(tuple) of features in different scales in the model
        # and the label and maybe it's downsampled outputs list(tuple) or some other supervisory signals
        for i in args:
            assert isinstance(
                i, (tuple, list)
            ), "all args must be either tuple or list, got %s" % type(i)

        if self.weight is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight

        # initialize the loss like this instead of 0 to ensure it sits on the correct device, not sure if that's
        # really necessary
        l = weights[0] * self.loss(*[j[0] for j in args])
        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                continue
            l += 0 if weights[i] == 0 else weights[i] * self.loss(*inputs)

        return l
