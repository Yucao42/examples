output = []
with open('result', 'r') as f:
    #w, h,   = map(float, f.readline().split('  '))
    tmp = []
    for i, line in enumerate(f):
        tmp.append(map(float, line.split('  ')))
        print(line.split('  '))
        output.append(tmp)
print(output)
