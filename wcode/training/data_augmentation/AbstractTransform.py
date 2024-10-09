import abc
from typing import List


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, keys: List) -> None:
        """
        keys: this transform needs to preprocess image, label or both.
        """
        assert isinstance(keys, List)
        assert set(self.keys).issubset({"image", "label"})
        self.keys = keys

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = (
            str(type(self).__name__)
            + "( "
            + ", ".join([key + " = " + repr(val) for key, val in self.__dict__.items()])
            + " )"
        )
        return ret_str
