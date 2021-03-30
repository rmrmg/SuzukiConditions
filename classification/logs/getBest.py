import subprocess, sys

minims=dict()

for fn in sys.argv[1:]:
    cmd = 'cat '+fn +' | grep "base: "'
    try:
        res = subprocess.check_output(cmd.encode(), shell=True)
    except:
        print("problem with ", fn)
        continue
    res = res.decode()
    res = res.split('\n')
    for re in res:
        if not re:
            continue
        epoch, _, basemean, basestd, _, solvmean, solvstd = re.split()
        basemean = round(float(basemean)*100,3)
        solvmean = round(float(solvmean)*100,3)
        basestd = round(float(basestd)*100,3)
        solvstd = round(float(solvstd)*100,3)
        summ = basemean+solvmean
        if not fn in minims:
            minims[fn] = [(basemean, basestd, solvmean, solvstd, epoch), (basemean, basestd, solvmean, solvstd, epoch), (summ, basemean, basestd, solvmean, solvstd,epoch)]
        if basemean > minims[fn][0][0]:
            minims[fn][0] = (basemean, basestd, solvmean, solvstd, epoch)
        if solvmean > minims[fn][1][2]:
            minims[fn][1] = (basemean, basestd, solvmean, solvstd,epoch)
        if summ >  minims[fn][2][0]:
            minims[fn][2] = (summ, basemean, basestd, solvmean, solvstd, epoch)
    #for x in minims[fn]:
    #    print(fn, x)
    print(fn, minims[fn][-1])