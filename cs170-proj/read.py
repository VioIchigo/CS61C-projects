def read_output(blabla):
    inp = "output/"+str(blabla)+".out"
    f = open(inp, 'r')
    x = f.readlines()
    f.close()
    p = x[0]
    path = []
    rec = ''
    for i in p:
        if i != ' ':
            rec += i
        else:
            path.append(int(rec))
            rec = ''
    print(path)

read_output(1)
