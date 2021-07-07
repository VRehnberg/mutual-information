import itertools
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import perfplot
import torch
import seaborn as sns
import pandas as pd
import opt_einsum

sns.set_style("whitegrid")

def yield_opts():
    yield from [
        "ab,ab->a",
        "ab,bc->ac",
        "ab,cd->ac",
    ]

def func(opt):
    return lambda t:opt_einsum.contract(opt, t, t)

r = perfplot.bench(
    setup=lambda n: torch.rand(n, n),
    kernels=[
        func(opt)
        for opt in yield_opts()
    ],
    labels=[
        opt for opt in yield_opts()
    ],
    n_range=[int(2**k) for k in np.linspace(11.5, 13, 10)],
    target_time_per_measurement=1.0,
    xlabel="n",
    equality_check=None,
)

plt.loglog()
r.plot()
plt.legend()

data = pd.DataFrame({
    label: [np.log(y2/y) / np.log(x2/x) for (x, y), (x2, y2) in itertools.combinations(zip(r.n_range, line), 2)]
    for line, label in zip(r.timings_s, r.labels)
})
sns.displot(data, kind="ecdf")

ax = plt.gca()
ax.grid(True)

plt.show()
