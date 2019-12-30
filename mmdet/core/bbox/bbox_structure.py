from collections import Sequence

import numpy as np


class BBox:
    """Assume the values of a bbox are [xmin, ymin, xmax, ymax] and of type float.
    We follow Python's slicing style, i.e. xmin:xmax contains pixels from xmin to xmax - 1, not including xmax.
    """

    _values_definition = ["xmin", "ymin", "xmax", "ymax"]

    def __init__(self, values):
        self.bbox = np.array(values, dtype=np.float32)

    def __getattr__(self, item):
        if item in self._values_definition:
            return self.bbox[self._values_definition.index(item)]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @classmethod
    def from_sequence(cls, bbox):
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            return BBox(bbox)
        raise ValueError(f"Invalid input for BBox: {bbox!r}")

    @classmethod
    def from_values(cls, *values):
        return cls.from_sequence(values)

    @classmethod
    def from_pct_repr(cls, bbox_pct_repr, im_size):
        """
        Initialize bbox from the percentage values taking a bigger image as reference.
        :param bbox_pct_repr: 4-value float between 0.0 and 1.0
        :param im_size: 2-value sequence: [width, height]
        :return:
        """
        return cls.from_sequence([c * s for c, s in zip(bbox_pct_repr, im_size * 2)])

    @property
    def values(self):
        return self.bbox.tolist()

    @property
    def int_values(self):
        return self.bbox.round().astype(np.int32).tolist()

    @property
    def center(self):
        """
        :return: 2-value numpy.ndarray, [x, y]
        """
        return np.array([(self.bbox[i+2] + self.bbox[i]) / 2 for i in (0, 1)])

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def w(self):
        return self.width

    @property
    def h(self):
        return self.height

    @property
    def c(self):
        return self.center
