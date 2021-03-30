import sys

lidx = 0
for line in open(sys.argv[1]):
    lidx +=1
    li  = line.split(' ')
    poz = 0
    mi=0
    ma=0
    #print( len(li), end=' ' )
    for elem in li:
        poz +=1
        try:
            v=float(elem)
            if v < mi:
                mi = v
            if v > ma:
                ma = v
        except:
            print("PROBLEM", lidx, poz, elem)
print(mi, ma)