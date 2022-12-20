import csv
import operator
from collections import defaultdict
from functools import reduce
import json


class nested_defaultdict:
    """Convenience class to create an arbitrary nested dictionary
    using defaultdict. The dictionary can be accessed using tuples
    of keys (k1,k2,k3,...).
    """

    def __init__(self):
        self._nested_dd = self.nested_default_dict()

    def __repr__(self):
        return json.dumps(self._nested_dd)

    def __str__(self):
        return json.dumps(self._nested_dd)

    def nested_default_dict(self):
        """Define a nested default dictionary"""
        return defaultdict(self.nested_default_dict)

    def get(self, map_list):
        """Get an item from a nested dictionary using a tuple of keys"""
        return reduce(operator.getitem, map_list, self._nested_dd)

    def set(self, map_list, value):
        """Set an item in a nested dictionary using a tuple of keys"""
        self.get(map_list[:-1])[map_list[-1]] = value

    def to_dict(self):
        return json.loads(json.dumps(self._nested_dd))


def csv_to_list(csv_path):
    file_list = []
    reader = csv.DictReader(open(csv_path))
    for i, line in enumerate(reader):
        file_list.append(line)
    return file_list


def iter_bids_dict(bids_dict: dict, _depth=0, max_depth=1):
    """Return a single iterator over the dictionary obtained from
    iter_dir - flexibly handles cases with and without a session date.
    Taken from https://thispointer.com/python-how-to-iterate-over-
    nested-dictionary-dict-of-dicts/
    """
    assert _depth >= 0
    for key, value in bids_dict.items():
        if isinstance(value, dict) and _depth < max_depth:
            # If value is dict then iterate over all its values
            for keyvalue in iter_bids_dict(
                value, _depth + 1, max_depth=max_depth
            ):
                yield (key, *keyvalue)
        else:
            # If value is not dict type then yield the value
            yield (key, value)
