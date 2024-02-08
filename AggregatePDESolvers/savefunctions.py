import numpy as np
import os


class SavingFunction:
    def __init__(self):
        pass

    def __call__(self, x=None):
        try:
            return np.load(self.to_path(x) + ".npy")
        except:
            value = self.get_value(x)
            if self.save_val(x):
                np.save(self.to_path(x) + ".npy", value)
            return value

    def __eq__(self, other):
        return self.to_path() == other.to_path()

    def make_dir(self):

        path = self.to_path()
        # remove the last part of the path
        path = path[: path.rfind("/")]
        if not os.path.exists(path):
            os.makedirs(path)


class Nothing(SavingFunction):
    def __init__(self):
        pass

    def to_path(self, x=None, safe=False):
        return ""


class ParametrizedSavingFunction(SavingFunction):
    def __init__(self, f, param_dic_hash, name, path, save_val=True):
        SavingFunction.__init__(self)
        self.param_dic_hash = param_dic_hash
        self.save = save_val
        self.name = name
        self.f = f
        self.path = path
        self.make_dir()

    def get_value(self, x):
        return self.f(x())

    def to_path(self, x=None, safe=False):
        if x is None:
            res = self.to_path(Nothing())
        else:
            res = f"{self.path}/{self.name}/{self.param_dic_hash}/({x.to_path(safe=True)})"
        if safe:
            return res.replace("/", "_").replace(".", "")
        return res

    def save_val(self, x):
        return self.save


class InputForSavingFunction(SavingFunction):
    def __init__(self, value, path, name="input", in_memory=True):
        SavingFunction.__init__(self)
        self.name = name
        self.in_memory = in_memory
        self.path = path
        if in_memory:
            self.value = value

        else:
            self.make_dir()
            np.save(self.to_path(), value)

    def get_value(self, x):
        try:
            return self.value
        except:
            raise ValueError(
                "ImputForSavingFunction should not be called when in_memory is False"
            )

    def save_val(self, x):
        return not self.in_memory

    def to_path(self, x=None, safe=False):
        if x is None:
            res = f"{self.path}/{self.name}"
            if safe:
                return res.replace("/", "_").replace(".", "")
            return res
        raise ValueError("ImputForSavingFunction should not be called with an argument")


class PDESolver(ParametrizedSavingFunction):
    def __init__(self, f, param_hash, name, path, save_val=True):
        ParametrizedSavingFunction.__init__(self, f, param_hash, name, path, save_val)

    def get_value(self, x):
        return self.f(x)
