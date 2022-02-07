import sys, statistics

#==> checkpoint-9614-epoch-19 MEAN 18.94225176629732
epochs = dict()
for fn in sys.argv[1:]:
    for line in open(fn):
        if not('TOPS' in line and 'epoch' in line):
            continue
        _, name, _, top1, top2, top3 = line.split()
        try:
            _, epoch = name.split('-epoch-')
        except:
            print("N", name, "LIN", line)
            raise

        top1 = float(top1)*100
        top2 =float(top2)*100
        top3 = float(top3)*100
        top2 += top1
        top3 += top2
        if not epoch in epochs:
            epochs[epoch]=([], [], [])
        epochs[epoch][0].append(top1)
        epochs[epoch][1].append(top2)
        epochs[epoch][2].append(top3)

for i in epochs:
    top1m = statistics.mean(epochs[i][0])
    top1s = statistics.stdev(epochs[i][0])

    top2m = statistics.mean(epochs[i][1])
    top2s = statistics.stdev(epochs[i][1])

    top3m = statistics.mean(epochs[i][2])
    top3s = statistics.stdev(epochs[i][2])

    print(i, top1m, top2m, top3m, top1s, top2s, top3s, sep='\t')