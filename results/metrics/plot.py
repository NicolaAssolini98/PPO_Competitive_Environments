import numpy as np
import matplotlib.pyplot as plt


dataName = "simple_tag"
#dataName = "simple_adversary"
#dataName = "simple_push"
#data = np.genfromtxt("simple_push_PPO_TD_maxstep200.csv", delimiter=",", names=["x", "y", "z"])
data = np.genfromtxt(dataName + "_maxstep400.csv", delimiter=",", names=["x", "y", "z"])
plt.plot(data['x'], data['y'])
f = plt.savefig(dataName + "GoodAgnt.png")
#plt.close(f)
mean = 0.0
means = []
means.append(0)
i = 0
for x in data['y']:
    if i > 0:
        mean = mean*i + float(x)
        mean /= i+1
        means.append(mean)
    i += 1
plt.plot(data['x'], means)
f = plt.savefig(dataName +"MeanGA.png")
plt.close(f)

mean = 0.0
means = []
means.append(0)
i = 0
for x in data['z']:
    if i > 0:
        mean = mean*i + float(x)
        mean /= i+1
        means.append(mean)
    i += 1

plt.plot(data['x'], data['z'])
f = plt.savefig(dataName + "BAgnt.png")
#plt.close(f)
plt.plot(data['x'], means)
f = plt.savefig(dataName +"MeanBA.png")
plt.close(f)

