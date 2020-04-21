import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('resultsMatrix.txt', converters = {0: lambda s: int(s[1:])})

# print(data[0:6, 1])
print(int(data.shape[0]/6))

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 1]
    b = data[i*6:i*6+6, 2]
    plt.bar(np.arange(1, 12, 2)+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
a = data[-1, 1]
b = data[-1, 2]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
labelIndexArray = np.arange(1, 14, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('stocks taken / stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', 'random'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 7]
    b = data[i*6:i*6+6, 8]
    plt.bar(np.arange(1, 12, 2)+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
a = data[-1, 7]
b = data[-1, 8]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
labelIndexArray = np.arange(1, 14, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('avg. stocks taken / avg. stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', 'random'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 3]
    b = data[i*6:i*6+6, 4]
    plt.bar(np.arange(1, 12, 2) + offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
a = data[-1, 3]
b = data[-1, 4]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
labelIndexArray = np.arange(1, 14, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('% \dealt / % \\received')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', 'random'])
plt.show()