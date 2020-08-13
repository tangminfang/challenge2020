import numpy as np
import matplotlib.pyplot as plt


acc_oval_temp=np.load("acc_oval_temp.npy")
f_oval_temp=np.load("f_oval_temp.npy")
Fbeta_oval_temp=np.load("Fbeta_oval_temp.npy")
Gbeta_oval_temp=np.load("Gbeta_oval_temp.npy")
auroc_temp=np.load("auroc_temp.npy")
auprc_temp=np.load("auprc_temp.npy")

plt.figure()
plt.plot(acc_oval_temp)
plt.show()
plt.figure()
plt.plot(f_oval_temp)
plt.show()
plt.figure()
plt.plot(Fbeta_oval_temp)
plt.show()
plt.figure()
plt.plot(Gbeta_oval_temp)
plt.show()
plt.figure()
plt.plot(auroc_temp)
plt.show()
plt.figure()
plt.plot(auprc_temp)
plt.show()