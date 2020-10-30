import math
import matplotlib.pyplot as plt

# open test result and store to list
r1 = open('test_result/top_cor_list.txt', 'r')
r2 = open('test_result/bot_cor_list.txt', 'r')
r3 = open('test_result/lung_size_list.txt', 'r')

top_cor_list = []
top_cor_list_split = []
bot_cor_list = []
bot_cor_list_split = []
lung_size_list = []
lung_size_list_split = []

while True:
    line = r1.readline()
    if not line:
        break
    top_cor_list.append(round(float(line), 2))
while True:
    line = r2.readline()
    if not line:
        break
    bot_cor_list.append(round(float(line), 2))
while True:
    line = r3.readline()
    if not line:
        break
    lung_size_list.append(round(float(line), 2))

# print total mean
top_cor_list_mean = sum(top_cor_list) / len(top_cor_list)
bot_cor_list_mean = sum(bot_cor_list) / len(bot_cor_list)
lung_size_list_mean = sum(lung_size_list) / len(lung_size_list)
print("total mean")
print("top coord: ", top_cor_list_mean)
print("bot coord: ", bot_cor_list_mean)
print("lung size: ", lung_size_list_mean)
print()

# print total standard deviation
print("total standard deviation")
print("top coord: ", math.sqrt(sum([(x - top_cor_list_mean) * (x - top_cor_list_mean) for x in top_cor_list]) / (len(top_cor_list) - 1)))
print("bot coord: ", math.sqrt(sum([(x - bot_cor_list_mean) * (x - bot_cor_list_mean) for x in bot_cor_list]) / (len(bot_cor_list) - 1)))
print("lung size: ", math.sqrt(sum([(x - lung_size_list_mean) * (x - lung_size_list_mean) for x in lung_size_list]) / (len(lung_size_list) - 1)))
print()

# print total size
print("total list size")
print("top coord: ", len(top_cor_list))
print("bot coord: ", len(bot_cor_list))
print("lung size: ", len(lung_size_list))
print()

# split list into 10 samples
for i in range(10):
    split = top_cor_list[int(i*len(top_cor_list)/10):int((i+1)*len(top_cor_list)/10)]
    top_cor_list_split.append(split)
    split = bot_cor_list[int(i*len(bot_cor_list)/10):int((i+1)*len(bot_cor_list)/10)]
    bot_cor_list_split.append(split)
    split = lung_size_list[int(i*len(lung_size_list)/10):int((i+1)*len(lung_size_list)/10)]
    lung_size_list_split.append(split)

# test top coordinate
top_cor_accuracy = []

for i in range(10):
    test = top_cor_list_split[i]
    train = []
    for j in range(10):
        if i != j:
            train.extend(top_cor_list_split[j])
    train_mean = sum(train) / len(train)
    new_test = [x - train_mean for x in test]
    test_mean = sum(new_test) / len(new_test)
    new_test_10mm = [x for x in new_test if x <= 10 and x >= -10]
    top_cor_accuracy.append(len(new_test_10mm) / len(new_test))
    print(len(new_test_10mm) / len(new_test))

print("top coord accuracy:", sum(top_cor_accuracy) / len(top_cor_accuracy))
print()

# test bot coordinate
bot_cor_accuracy = []

for i in range(10):
    test = bot_cor_list_split[i]
    train = []
    for j in range(10):
        if i != j:
            train.extend(bot_cor_list_split[j])
    train_mean = sum(train) / len(train)
    new_test = [x - train_mean for x in test]
    test_mean = sum(new_test) / len(new_test)
    new_test_10mm = [x for x in new_test if x <= 10 and x >= -10]
    bot_cor_accuracy.append(len(new_test_10mm) / len(new_test))
    print(len(new_test_10mm) / len(new_test))

print("bot coord accuracy:", sum(bot_cor_accuracy) / len(bot_cor_accuracy))
print()

# test lung size
lung_size_accuracy = []

for i in range(10):
    test = lung_size_list_split[i]
    train = []
    for j in range(10):
        if i != j:
            train.extend(lung_size_list_split[j])
    train_mean = sum(train) / len(train)
    new_test = [x - train_mean for x in test]
    test_mean = sum(new_test) / len(new_test)
    new_test_10mm = [x for x in new_test if x <= 20 and x >= -20]
    lung_size_accuracy.append(len(new_test_10mm) / len(new_test))
    print(len(new_test_10mm) / len(new_test))

print("lung size accuracy:", sum(lung_size_accuracy) / len(lung_size_accuracy))
print()


# plot test result
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].set_ylim(-30, 30)
ax[1].set_ylim(-30, 30)
ax[0].set_ylabel("predict value - answer(mm)")
ax[0].set_xlabel("image index")
ax[1].set_ylabel("predict value - answer(mm)")
ax[1].set_xlabel("image index")
ax[0].set_title("top coordinate")
ax[1].set_title("bottom coordinate")

ax[0].axhline(y=0, color='r', linewidth=0.5)
ax[0].axhline(y=-15, color='r', linewidth=0.05)
ax[0].axhline(y=15, color='r', linewidth=0.05)
ax[0].axhline(y=-10, color='r', linewidth=0.2)
ax[0].axhline(y=10, color='r', linewidth=0.2)
ax[0].axhline(y=-5, color='r', linewidth=0.35)
ax[0].axhline(y=5, color='r', linewidth=0.35)

ax[1].axhline(y=0, color='r', linewidth=0.5)
ax[1].axhline(y=-15, color='r', linewidth=0.05)
ax[1].axhline(y=15, color='r', linewidth=0.05)
ax[1].axhline(y=-10, color='r', linewidth=0.2)
ax[1].axhline(y=10, color='r', linewidth=0.2)
ax[1].axhline(y=-5, color='r', linewidth=0.35)
ax[1].axhline(y=5, color='r', linewidth=0.35)


test = top_cor_list_split[0]
train = []
for j in range(1, 10):
    train.extend(top_cor_list_split[j])
train_mean = sum(train) / len(train)
new_test = [x - train_mean for x in test]

xs = [x for x in range(len(new_test))]
ax[0].plot(xs, new_test, '.', markersize=5)


test = bot_cor_list_split[0]
train = []
for j in range(1, 10):
    train.extend(bot_cor_list_split[j])
train_mean = sum(train) / len(train)
new_test = [x - train_mean for x in test]
ax[1].plot(xs, new_test, '.', markersize=5)

plt.savefig('result.png')
