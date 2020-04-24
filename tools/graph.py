import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('resultsMatrix.txt', converters = {0: lambda s: int(s[1:])})

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 1]
    b = data[i*6:i*6+6, 2]
    plt.bar(np.arange(1, 12, 2)+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
# a = data[-4, 1]
# b = data[-4, 2]
a = data[38, 1]
b = data[38, 2]
plt.bar(5+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[39, 1]
b = data[39, 2]
plt.bar(9+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6, 1]
b = data[int(data.shape[0]/6)*6, 2]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6+1, 1]
b = data[int(data.shape[0]/6)*6+1, 2]
plt.bar(15, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

labelIndexArray = np.arange(1, 16, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('stocks taken / stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', '32:128:256 w/o recovery', '16:32:32 w/o recovery', 'random w/ recovery', 'random w/o recovery'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 7]
    b = data[i*6:i*6+6, 8]
    plt.bar(np.arange(1, 12, 2)+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25

a = data[38, 7]
b = data[38, 8]
plt.bar(5+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[39, 7]
b = data[39, 8]
plt.bar(9+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6, 7]
b = data[int(data.shape[0]/6)*6, 8]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6+1, 7]
b = data[int(data.shape[0]/6)*6+1, 8]
plt.bar(15, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

labelIndexArray = np.arange(1, 16, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('avg. stocks taken / avg. stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', '32:128:256 w/o recovery', '16:32:32 w/o recovery', 'random w/ recovery', 'random w/o recovery'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 3]
    b = data[i*6:i*6+6, 4]
    plt.bar(np.arange(1, 12, 2) + offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25

a = data[38, 3]
b = data[38, 4]
plt.bar(5+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[39, 3]
b = data[39, 4]
plt.bar(9+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6, 3]
b = data[int(data.shape[0]/6)*6, 4]
plt.bar(13, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

a = data[int(data.shape[0]/6)*6+1, 3]
b = data[int(data.shape[0]/6)*6+1, 4]
plt.bar(15, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)

labelIndexArray = np.arange(1, 16, 2)
plt.xticks(labelIndexArray, ['1', '3', '5', '7', '9', '11', 'none', 'none'])
# plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('% \dealt / % \\received')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64', '32:128:256', '32:128:256 w/o recovery', '16:32:32 w/o recovery', 'random w/ recovery', 'random w/o recovery'])
plt.show()