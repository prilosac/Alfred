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
plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('stocks taken / stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 7]
    b = data[i*6:i*6+6, 8]
    plt.bar(np.arange(1, 12, 2)+offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('avg. stocks taken / avg. stocks lost')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64'])
plt.show()

offset = 0
for i in range(int(data.shape[0]/6)):
    a = data[i*6:i*6+6, 3]
    b = data[i*6:i*6+6, 4]
    plt.bar(np.arange(1, 12, 2) + offset, np.divide(a, b, out=np.zeros_like(a), where=b!=0), width=0.25)
    offset += 0.25
plt.xticks(np.arange(1, 12, 2))
plt.xlabel('kernel size')
plt.ylabel('% \dealt / % \\received')
plt.legend(labels=['16:32:32', '16:64:256', '32:64:64', '16:64:64', '16:32:64'])
plt.show()


# plt.legend(labels=['LR Classification', 'ROLR Classification', 'LinearSVC', 'Decision Tree', 'kNN'])
# # plt.title(r"$\sigma_o = 10$")
# plt.show()

# plt.plot(np.arange(0.0, 1.3, 0.1), lr_error)
# plt.plot(np.arange(0.0, 1.3, 0.1), lrp_error)
# plt.plot(np.arange(0.0, 1.3, 0.1), rolr_error)
# plt.plot(np.arange(0.0, 1.3, 0.1), linear_svr_error)
# plt.plot(np.arange(0.0, 1.3, 0.1), linear_svr_p_error)
# plt.xlabel('outlier to inlier ratio')
# plt.ylabel('error: ' + r"$||\^\beta - \beta^*||$")
# plt.xlim([0,1.2])
# plt.legend(labels=['LR', 'LR+P', 'ROLR', 'LinearSVR', 'LinearSVR+P'])
# # plt.title(r"$\sigma_o = 10$")
# plt.show()