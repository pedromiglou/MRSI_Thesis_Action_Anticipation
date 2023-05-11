import random

f = open("partial_data.csv", "w")

for j in range(10):
    for i in range(13):
        num = (i*3)%10
        f.write(f"{j},{j},{num},")
    f.write(f"{j}\n")

f.close()