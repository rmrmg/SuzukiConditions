import subprocess, sys

minims=dict()

for fn in sys.argv[1:]:
    for line in open(fn):
        epoch, _, mean, std = line.split()
        mean = round(float(mean)*100,3)
        std = round(float(std)*100,3)
        if (not fn in minims) or (mean < minims[fn][0]):
            minims[fn] = (mean, std, epoch)

    print(fn, minims[fn])