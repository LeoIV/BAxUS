import os
import stat
import subprocess
import sys
import tempfile
import urllib
from logging import info, warning
from platform import machine
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from baxus.benchmarks import SyntheticBenchmark, EffectiveDimBenchmark


class MoptaSoftConstraints(SyntheticBenchmark):
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(
            self,
            temp_dir: Optional[str] = None,
            binary_path: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        super().__init__(124, np.ones(124), np.zeros(124), noise_std=noise_std)
        if binary_path is None:
            self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
            self.machine = machine().lower()
            if self.machine == "armv7l":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_armhf.bin"
            elif self.machine == "x86_64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_elf64.bin"
            elif self.machine == "i386":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_elf32.bin"
            elif self.machine == "amd64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_amd64.exe"
            else:
                raise RuntimeError("Machine with this architecture is not supported")
            self._mopta_exectutable = os.path.join(
                os.getcwd(), "baxus", "benchmarks", "mopta08", self._mopta_exectutable
            )

            if not os.path.exists(self._mopta_exectutable):
                basename = os.path.basename(self._mopta_exectutable)
                info(f"Mopta08 executable for this architecture not locally available. Downloading '{basename}'...")
                urllib.request.urlretrieve(
                    f"https://mopta.papenmeier.io/{os.path.basename(self._mopta_exectutable)}",
                    self._mopta_exectutable)
                os.chmod(self._mopta_exectutable, stat.S_IXUSR)

        else:
            self._mopta_exectutable = binary_path
        if temp_dir is None:
            self.directory_file_descriptor = tempfile.TemporaryDirectory()
            self.directory_name = self.directory_file_descriptor.name
        else:
            if not os.path.exists(temp_dir):
                warning(f"Given directory '{temp_dir}' does not exist. Creating...")
                os.mkdir(temp_dir)
            self.directory_name = temp_dir

    def __call__(self, x):
        super(MoptaSoftConstraints, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        # create tmp dir for mopta binary

        vals = np.array([self._call(y) for y in x]).squeeze()
        return vals + np.random.normal(
            np.zeros_like(vals), np.ones_like(vals) * self.noise_std, vals.shape
        )

    def _call(self, x: np.ndarray):
        """
        Evaluate Mopta08 benchmark for one point

        Args:
            x: one input configuration

        Returns:value with soft constraints

        """
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """
        Return the "optimal" value.

        Returns:
            np.ndarray: -200, some guessed optimal value we never beat

        """
        return np.array(-200.0)


class LassoLeukemiaBenchmark(EffectiveDimBenchmark):
    """
    7129-D Leukemia benchmark from https://github.com/ksehic/LassoBench

    Args:
        noise_std: ignored
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="leukemia", mf_opt="discrete_fidelity"
        )
        dim = self._b.n_features

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=22,
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        )


class LassoBreastCancerBenchmark(EffectiveDimBenchmark):
    """
    10-D breast cancer benchmark from https://github.com/ksehic/LassoBench

    Args:
        noise_std: ignored
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="breast_cancer", mf_opt="discrete_fidelity"
        )
        dim = self._b.n_features

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=3,
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        )


class LassoDiabetesBenchmark(EffectiveDimBenchmark):
    """
   8-D diabetes benchmark from https://github.com/ksehic/LassoBench

   Args:
       noise_std: ignored
       **kwargs:
   """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="diabetes", mf_opt="discrete_fidelity"
        )
        dim = self._b.n_features

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=5,
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        )


class LassoDNABenchmark(EffectiveDimBenchmark):
    """
    180-D DNA benchmark from https://github.com/ksehic/LassoBench

    Args:
        noise_std: ignored
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="dna", mf_opt="discrete_fidelity"
        )
        dim = self._b.n_features

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=43,
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        )


class LassoRCV1Benchmark(EffectiveDimBenchmark):
    """
    19 959-D RCV1 benchmark from https://github.com/ksehic/LassoBench

    Args:
        noise_std: ignored
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="rcv1", mf_opt="discrete_fidelity"
        )
        dim = self._b.n_features

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=75,
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        )


class LassoSimpleBenchmark(EffectiveDimBenchmark):
    """
    60-D synthetic Lasso simple benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        noise_std: if > 0: noisy version with fixed SNR, noiseless version otherwise
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):

        from LassoBench import LassoBench

        if noise_std > 0:
            warning(
                f"LassoBenchmark with noise_std {noise_std} chosen. Will use noisy version with snr ratio 10. The exact value of noise_std will be ignored."
            )
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_simple", noise=noise_std > 0
        )
        dim = self._b.n_features

        self.effective_dims = np.arange(dim)[self._b.w_true != 0]
        info(f"function effective dimensions: {self.effective_dims.tolist()}")

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=len(self.effective_dims),
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        return np.array(result_list).squeeze()


class LassoMediumBenchmark(EffectiveDimBenchmark):
    """
    100-D synthetic Lasso medium benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        noise_std: if > 0: noisy version with fixed SNR, noiseless version otherwise
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):
        from LassoBench import LassoBench

        if noise_std > 0:
            warning(
                f"LassoBenchmark with noise_std {noise_std} chosen. Will use noisy version with snr ratio 10. The exact value of noise_std will be ignored."
            )
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_medium", noise=noise_std > 0
        )
        dim = self._b.n_features

        self.effective_dims = np.arange(dim)[self._b.w_true != 0]
        info(f"function effective dimensions: {self.effective_dims.tolist()}")

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=len(self.effective_dims),
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        return np.array(result_list).squeeze()


class LassoHighBenchmark(EffectiveDimBenchmark):
    """
    300-D synthetic Lasso high benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        noise_std: if > 0: noisy version with fixed SNR, noiseless version otherwise
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):
        from LassoBench import LassoBench

        if noise_std > 0:
            warning(
                f"LassoBenchmark with noise_std {noise_std} chosen. Will use noisy version with snr ratio 10. The exact value of noise_std will be ignored."
            )
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_high", noise=noise_std > 0
        )
        dim = self._b.n_features

        self.effective_dims = np.arange(dim)[self._b.w_true != 0]
        info(f"function effective dimensions: {self.effective_dims.tolist()}")

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=len(self.effective_dims),
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        return np.array(result_list).squeeze()


class LassoHardBenchmark(EffectiveDimBenchmark):
    """
    1000-D synthetic Lasso hard benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        noise_std: if > 0: noisy version with fixed SNR, noiseless version otherwise
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):
        from LassoBench import LassoBench

        if noise_std > 0:
            warning(
                f"LassoBenchmark with noise_std {noise_std} chosen. Will use noisy version with snr ratio 10. The exact value of noise_std will be ignored."
            )
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_hard", noise=noise_std > 0
        )
        dim = self._b.n_features

        self.effective_dims = np.arange(dim)[self._b.w_true != 0]
        info(f"function effective dimensions: {self.effective_dims.tolist()}")

        super().__init__(
            dim=dim,
            ub=np.full(dim, fill_value=1.0),
            lb=np.full(dim, fill_value=-1.0),
            effective_dim=len(self.effective_dims),
            noise_std=noise_std,
        )

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        return np.array(result_list).squeeze()


class SVMBenchmark(SyntheticBenchmark):
    def __init__(
            self,
            data_folder: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        """
        SVM Benchmark from https://arxiv.org/abs/2103.00349

        Support also a noisy version where the model is trained on random subset of 250 points
        which is used whenever noise_std is greater than 0.

        Args:
            data_folder: the folder where the slice_localization_data.csv is located
            noise_std: noise standard deviation. Anything greater than 0 will lead to a noisy benchmark
            **kwargs:
        """
        self.value = np.inf
        self.best_config = None
        self.noisy = noise_std > 0
        if self.noisy:
            warning("Using a noisy version of SVMBenchmark where training happens on a random subset of 250 points."
                    "However, the exact value of noise_std is ignored.")
        super(SVMBenchmark, self).__init__(
            388, lb=np.zeros(388), ub=np.ones(388), noise_std=noise_std
        )
        self.X, self.y = self._load_data(data_folder)
        if not self.noisy:
            np.random.seed(388)
            idxs = np.random.choice(np.arange(len(self.X)), min(10000, len(self.X)), replace=False)
            half = len(idxs) // 2
            self._X_train = self.X[idxs[:half]]
            self._X_test = self.X[idxs[half:]]
            self._y_train = self.y[idxs[:half]]
            self._y_test = self.y[idxs[half:]]

    def _load_data(self, data_folder: Optional[str] = None):
        if data_folder is None:
            data_folder = os.path.join(os.getcwd(), "data")
        if not os.path.exists(os.path.join(data_folder, "CT_slice_X.npy")):
            sld_dir = os.path.join(data_folder, "slice_localization_data.csv.xz")
            sld_bn = os.path.basename(sld_dir)
            info(f"Slice localization data not locally available. Downloading '{sld_bn}'...")
            urllib.request.urlretrieve(
                f"http://mopta-executables.s3-website.eu-north-1.amazonaws.com/{sld_bn}",
                sld_dir)
            data = pd.read_csv(
                os.path.join(data_folder, "slice_localization_data.csv.xz")
            ).to_numpy()
            X = data[:, :385]
            y = data[:, -1]
            np.save(os.path.join(data_folder, "CT_slice_X.npy"), X)
            np.save(os.path.join(data_folder, "CT_slice_y.npy"), y)
        X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
        y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        return X, y

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        super(SVMBenchmark, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        x = x ** 2

        errors = []
        for y in x:
            C = 0.01 * (500 ** y[387])
            gamma = 0.1 * (30 ** y[386])
            epsilon = 0.01 * (100 ** y[385])
            length_scales = np.exp(4 * y[:385] - 2)

            svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
            if self.noisy:
                np.random.seed(None)
                idxs = np.random.choice(np.arange(len(self.X)), min(500, len(self.X)), replace=False)
                half = len(idxs) // 2
                X_train = self.X[idxs[:half]]
                X_test = self.X[idxs[half:]]
                y_train = self.y[idxs[:half]]
                y_test = self.y[idxs[half:]]
                svr.fit(X_train / length_scales, y_train)
                pred = svr.predict(X_test / length_scales)
                error = np.sqrt(np.mean(np.square(pred - y_test)))
            else:
                svr.fit(self._X_train / length_scales, self._y_train)
                pred = svr.predict(self._X_test / length_scales)
                error = np.sqrt(np.mean(np.square(pred - self._y_test)))

            errors.append(error)
            if errors[-1] < self.value:
                self.best_config = np.log(y)
                self.value = errors[-1]
        return np.array(errors).squeeze()
