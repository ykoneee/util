import re
from pathlib import Path

import numpy as np
from scipy import stats

pt = Path("lightning_logs")


def kde_maxima(x):
    kde = stats.gaussian_kde(x)
    samples = np.linspace(min(x) - 0.2, max(x) + 0.2, 1000)

    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]
    return maxima


def ci(x):
    mu = np.mean(x)
    sigma = np.std(x)
    ci0, ci1 = stats.norm.interval(0.95, loc=mu, scale=sigma)
    return f"{mu:.3f}+-{(ci1 - ci0) / 2:.3f}"


pattern = re.compile(r".*0\.(\d+)-epoch.*")
result = []
result2 = []
result3 = []
for i in pt.glob("*"):
    i2 = i / "checkpoints"
    assert i2.exists()

    vmax = []
    for j in i2.glob("*.ckpt"):
        k = re.findall(pattern, j.name)
        if len(k) == 0:
            continue
        vmax.append(float("0." + k[0]))

    vmax1 = max(sorted(vmax))
    vmax2 = max(sorted(vmax)[:-1])
    vmax3 = max(sorted(vmax)[:-2])
    result.append(vmax1)
    result2.append(vmax2)
    result3.append(vmax3)

result = sorted(result)
result2 = sorted(result2)
result3 = sorted(result3)
length = len(result)
hlen = length // 2
print(f"{result} {length}")
print(f"{result2} ")
print(f"{result3} ")

print("origin result:===============================")
print(f"{ci(result)} kde: {kde_maxima(result):.3f}")
print(f"{ci(result2)} kde: {kde_maxima(result2):.3f}")
print(f"{ci(result3)} kde: {kde_maxima(result3):.3f}")
# print("origin result fake:===============================")
# print(f"{ci(result[hlen:])} kde: {kde_maxima(result[hlen:]):.3f}")
# print(f"{ci(result2[hlen:])} kde: {kde_maxima(result2[hlen:]):.3f}")
# print(f"{ci(result3[hlen:])} kde: {kde_maxima(result3[hlen:]):.3f}")
result = result[1:-1]
result2 = result2[1:-1]
result3 = result3[1:-1]
print("1:-1 result:===============================")
print(f"{ci(result)} kde: {kde_maxima(result):.3f}")
print(f"{ci(result2)} kde: {kde_maxima(result2):.3f}")
print(f"{ci(result3)} kde: {kde_maxima(result3):.3f}")

# print("1:-1 result fake:===============================")
# print(f"{ci(result[hlen:])} kde: {kde_maxima(result[hlen:]):.3f}")
# print(f"{ci(result2[hlen:])} kde: {kde_maxima(result2[hlen:]):.3f}")
# print(f"{ci(result3[hlen:])} kde: {kde_maxima(result3[hlen:]):.3f}")
