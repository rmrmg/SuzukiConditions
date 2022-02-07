import sys, statistics

#==> checkpoint-9614-epoch-19 MEAN 18.94225176629732
epochs = dict()
for fn in sys.argv[1:]:
    for line in open(fn):
        if not('MEAN' in line and 'epoch' in line):
            continue
        _, name, _, mae = line.split()
        try:
            _, epoch = name.split('-epoch-')
        except:
            print("N", name, "LIN", line)
            raise
        epoch = int(epoch)
        mae = float(mae)
        if not epoch in epochs:
            epochs[epoch]=[]
        epochs[epoch].append(mae)


for i in epochs:
    print(i, statistics.mean(epochs[i]), statistics.stdev(epochs[i]), epochs[i], sep='\t')