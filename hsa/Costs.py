from neon.transforms.cost import Cost


class SumSquaredWithL1Loss(Cost):
    def smoothL1(self, x):
        """
        Returns the Smooth-L1 cost
        """
        return (0.5 * self.be.square(x) * (self.be.absolute(x) < 1) +
                (self.be.absolute(x) - 0.5) * (self.be.absolute(x) >= 1))

    def smoothL1grad(self, x):
        """
        Returns the gradient of the Smooth-L1 cost.
        """
        return (x * (self.be.absolute(x) < 1) + self.be.sgn(x) *
                (self.be.absolute(x) >= 1))

    def __init__(self):
        """
        Define the cost function and its gradient as lambda functions.
        """
        # Sum squared
        super().__init__()
        self.func = lambda y, t: self.be.sum(self.be.square(y - t), axis=0) / 2. + self.be.sum(self.smoothL1(y - t),
                                                                                               axis=0)
        self.funcgrad = lambda y, t: (y - t) + self.smoothL1grad(y - t)
