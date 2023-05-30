flags = "Barbados,blue,orange,blue\nCameroon,green,red,yellow\nFrance,blue,white,red\nGuatemala,lightblue,white,lightblue\nGuinea,red,yellow,green\nIreland,green,white,orange\nItaly,green,white,red\nIvory_Coast,orange,white,green\nMali,green,yellow,red\nMongolia,red,blue,red\nPeru,red,white,red\nPortugal,green,red,red\nRomania,blue,yellow,red\nSt._Vincent_Grenadines,blue,yellow,green"

flags = flags.split("\n")

d = dict()

for f in flags:
    f = f.replace("blue", "dark_blue")
    
    f = f.replace("lightdark_blue", "light_blue")

    f = f.split(",")[1:]

    if (f[0],) not in d:
        d[(f[0],)] = set([f[1]])
    else:
        d[(f[0],)].add(f[1])

    if (f[0],f[1]) not in d:
        d[(f[0],f[1])] = set([f[2]])
    else:
        d[(f[0],f[1])].add(f[2])

f = open("new_rules.json","w")
f.write("[\n")

for k, v in d.items():
    k = str(list(k)).replace("'",'"')

    previous = '""'
    for next in v:
        f.write("\t{\n")
        f.write(f'\t\t"blocks": {k},\n')
        f.write(f'\t\t"refused": {previous},\n')
        previous = f'"{next}"'
        f.write(f'\t\t"next_block": "{next}"\n')
        f.write("\t},\n")

f.write("]")