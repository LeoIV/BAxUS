# Reproducing the results from the paper

We explain here how to reproduce the `BAxUS` and the `EmbeddedTuRBO` results from our paper.
Please increase the number of repetitions `-r` if too low.

## Main paper results

### Figure 2

For Figure 2, we used the following code:

```python
import math
import numpy as np
from matplotlib import pyplot as plt
```

Define the HeSBO success probability:
```python
def succ_hesbo(D,d,de):
    """
    HeSBO success probability (independent of D)
    """
    if d<de:
        return 0
    return np.math.factorial(d)/(np.math.factorial(d-de)*d**de)
```
Define the BAxUS success probability:
```python
def succ_baxus(D,d,de):
    """
    BAxUS success probability
    """
    if d<de:
        return 0
    beta_s = D//d
    beta_l = math.ceil(D/d)
    dl = D-d*beta_s
    ds = d-dl

    numerator = sum(math.comb(ds,i)*math.comb(dl,de-i)*beta_s**i*beta_l**(de-i) for i in range(de+1))
    return numerator/math.comb(D,de)
```
Plot:
```python
fig, axs = plt.subplots(1,3, figsize=(7.5,3.2), sharey=True)

plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

for p, D in enumerate([100,500,1000]):
    for o, d in enumerate([20]):
        ax = axs[p]
        hesbos = [succ_hesbo(D,i,d) for i in range(d,D+1)]
        baxuss = [succ_baxus(D,i,d) for i in range(d,D+1)]
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.plot(np.arange(d,len(hesbos)+d), baxuss, color="#E66100", marker="*", markevery=(D-d)//4, label=r"BAxUS")
        ax.plot(np.arange(d,len(hesbos)+d), hesbos, color="#5D3A9B", marker="^", markevery=(D-d)//4, label=r"HeSBO")
        
        ax.set_title(f"$D$={D}, $d_e$={d}")
        if o == 0:
            ax.set_xlabel(r"$d$", fontsize=12)
        if p == 0:
            ax.set_ylabel(r"$p(Y^*; D,d,d_e)$", fontsize=12)
handles, labels = axs[0].get_legend_handles_labels()
fig.tight_layout()
plt.subplots_adjust(bottom=0.35)
fig.legend(handles, labels, loc='lower center', ncol=2, fancybox=True, borderaxespad=2, columnspacing=3,fontsize="large")
```


### Figure 3
To reproduce the results from Figure 3, run

```bash
python benchmark_runner.py -a ALGORITHM -f BENCHMARK -m 1000 -r 20 -id 500 -td 1 --adjust-initial-target-dimension --n-init 10
```

where you replace `BENCHMARK` with `mopta08`, `svm`, `branin2`, `hartmann6`, `lasso-hard`, `lasso-high` and `ALGORITHM` with `baxus` or `random_search`.
For the Lasso benchmarks, replace `-m 1000` with `-m 2000` and add `--budget-until-input-dim 1000`.

This runs BAxUS or random search on the respective benchmark for 20 iterations. 
The input dimensionality is set to 500 but is overriden for all benchmarks having a fixed input dimensionality (here, all benchmarks except for `branin2` and `hartmann6`).

### Figure 4
To reproduce the results from Figure 4, run
```bash
python benchmark_runner.py -a baxus -f BENCHMARK -m 1000 -r 20 -id 500 -td 1 --adjust-initial-target-dimension --noise-std 1.0 --n-init 10
```

This runs the noisy version of Lasso Hard and Lasso High. 
For these benchmarks, the exact value of `--noise-std` is ignored; if it is greater than zero, the noisy version is run.

## Appendix results 

### Figure 5

To reproduce the results from Figure 5, run
```bash
python benchmark_runner.py -a embedded_turbo_target_dim -f shiftedackley10 -m 1000 -r 100 -id 30 -td 20 --embedding-type EMBEDDING_TYPE --n-init 10
```

where you replace `EMBEDDING_TYPE` with `baxus` or `hesbo`.

### Figure 6

For Figure 6, we used the following code

```python
from baxus.util.projections import AxUS
from baxus.util.behaviors.embedding_configuration import EmbeddingType
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import scipy

def important_bins(S, de):
    """
    Number of target bins containing important input dimensions.
    Assumes w.l.o.g. that the first de dimensions are the important input dimensions.
    """
    return len(set(v for k,v in S.input_to_target_dim.items() if k < de))

REPETITIONS = 50
EVALS = 100
D = 30
d = 20
de = 10

bar_width = 0.4

fig, ax = plt.subplots(1,1, figsize=(4.5,4))


for bs in [EmbeddingType.HESBO, EmbeddingType.AXUS]:
    group = -bar_width/2 if bs == EmbeddingType.HESBO else bar_width/2
    n_bins = defaultdict(int)
    histbinss = []
    for r in tqdm(range(REPETITIONS)):
        n_bins = defaultdict(int)
        for _ in range(EVALS):
            S = AxUS(input_dim=D, target_dim=d,bin_sizing=bs, seed=_)
            bins = important_bins(S,de)
            n_bins[bins] = n_bins[bins]+1
        sortkeys = np.arange(1,de+1)
        histbins = np.array([n_bins[k] for k in sortkeys])/(EVALS)

        histbinss.append(histbins)
    mean = np.mean(np.array(histbinss), axis=0)
    stderr = scipy.stats.sem(np.array(histbinss)[:,mean > 0], axis=0)
    print(mean.shape)
    sortkeys = np.array([i+1 for i in range(len(mean)) if mean[i]>0])
    hist_bins_format = [f"{f'{hb:.2f}'.replace('0.','.')}" for hb in mean]
    print("sc",sortkeys)

    color = "#5D3A9B" if bs == EmbeddingType.HESBO else "#E66100"
    label = "HeSBO" if bs == EmbeddingType.HESBO else "BAxUS"
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    bar = ax.bar(sortkeys+group, mean[mean > 0], yerr=stderr, width=bar_width, label=label, color=color)
    ax.bar_label(bar,  fmt='%.2f')
ax.set_xlabel("\# target dims. containing an important input dim.", size=14)
ax.set_ylabel("empirical probability", size=14)
ax.legend(loc="upper left")
fig.tight_layout()
```

### Figure 7

To reproduce the results from Figure 7, run
```bash
python benchmark_runner.py -a baxus -f lasso-hard -m 1000 -r 20 -id 500 -td 1 --adjust-initial-target-dimension --n-init 10
```
for the BAxUS result and 
```bash
python benchmark_runner.py -a embedded_turbo_target_dim -f lasso-hard -m 1000 -r 20 -id 500 -td TARGET_DIMENSION --n-init 10
```
where you replace `TARGET_DIMENSION` with 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, and 100.

### Figure 8
To reproduce the results from Figure 8, run
```bash
python benchmark_runner.py -a ALGORITHM -f lasso-dna -m 1000 -r 20 -id 500 -td 1 --adjust-initial-target-dimension --n-init 10
```
where you replace `ALGORITHM` with `baxus` or `random_search`.