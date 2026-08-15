from collections import OrderedDict

class PrefixLRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dict_data = OrderedDict()
        self.size_ = 0

    def query(self, key: int) -> int:
        return self.get(key)

    def get(self, key: int) -> int:
        if key in self.dict_data:
            res = self.dict_data[key]
            self.dict_data.move_to_end(key)
            return res
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dict_data:
            self.dict_data[key] = value
            self.dict_data.move_to_end(key)
        else:
            if self.size_ >= self.capacity:
                raise KeyError(f'queue is full, capacity is {self.capacity}')
            self.dict_data[key] = value
            self.dict_data.move_to_end(key)
            self.size_ += 1

    def pop(self):
        if self.size_ == 0:
            return None, None
        self.size_ -= 1
        return self.dict_data.popitem(last=False)

    def get_len(self) -> int:
        return self.size_