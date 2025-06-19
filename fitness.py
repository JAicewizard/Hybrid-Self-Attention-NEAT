import os
import csv
import numpy as np
import matplotlib.pyplot as plt

BASEDIR = "results"

def parse(path: str):
    avgs = []
    stds = []
    bests = []
    apples = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            avgs.append(float(row[3]))
            stds.append(float(row[4]))
            bests.append(float(row[5]))
            apples.append(float(row[6]))
    return avgs, stds, bests, apples


generations = list(range(100))

dirfig, dirax = plt.subplots(1, figsize=(10, 6))
dirfig.suptitle(f'Apples collected for best Fn over time, using the directional setup')
dirfig.supxlabel('Generation')
dirfig.supylabel('Apples collected')
dirax.set_xlim(0,100)
dirax.set_ylim(-1,40)
dirax.set_xticks(range(0,100, 5))
for i in range(0,5):
    fig, (avgax, bestax, applesax) = plt.subplots(3, figsize=(6, 8))
    fig.suptitle(f'Fitness over time using F{i}')
    fig.supxlabel('Generation')
    fig.supylabel('Fitness') # TODO this needs to be per axis
    avgax.set_title(f"Mean F{i}")
    bestax.set_title(f"Best F{i}")
    applesax.set_title(f"Apples collected for best F{i}")
    for experiment in os.listdir(BASEDIR):
        avgs, stds, bests, apples = parse(f"{BASEDIR}/{experiment}/training_log_{i}.csv")
        line, = avgax.plot(avgs)
        # line, = avgax.plot(avgs, linestyle='dotted')
        line.set_label(f"{experiment}")
        line, = bestax.plot(bests, color=line.get_color())
        line.set_label(f"{experiment}")
        line, = applesax.plot(apples, color=line.get_color())
        line.set_label(f"{experiment}")

        if experiment == "directional":
            line, = dirax.plot(apples, linewidth=1)
            line.set_label(f"Fitness = F{i}")
        # avgax.fill_between(generations, [i1 + i2 for i1, i2 in zip(avgs , stds)], [i1 - i2 for i1, i2 in zip(avgs , stds)], alpha=0.4)

    avgax.legend()
    bestax.legend()
    applesax.legend()
    fig.tight_layout()
    fig.savefig(f"results_{i}.png", dpi=300)

dirax.legend()
dirfig.tight_layout()
dirfig.savefig(f"directional.png", dpi=300)
