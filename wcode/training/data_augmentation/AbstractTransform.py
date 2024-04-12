import abc


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):

        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = (str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val)
             for key, val in self.__dict__.items()]) + " )")
        return ret_str
