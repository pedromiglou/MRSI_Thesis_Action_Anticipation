import json

pos = [-0.360, 0.543, 1.109, 0.695, 0.719, 0.009, -0.007] # above red 1

x = -0.13
y = 0.14
z = -0.08

blocks = ["red1", "red2", "dark_blue1", "dark_blue2", "light_blue1", "light_blue2", "yellow", "green", "orange", "white"]

d = dict()

for i in range(5):
    for j in range(2):
        name = blocks[0]
        blocks = blocks[1:]
        for k in range(2):
            new_pos = [round(pos[0]+j*x, 3), round(pos[1]+i*y, 3), round(pos[2]+k*z, 3), pos[3], pos[4], pos[5], pos[6]]
            print(new_pos)

            if k==0:
                d["above_"+name] = new_pos
            else:
                d[name] = new_pos

f = open("positions.json", "w")
f.write(json.dumps(d))
f.close()