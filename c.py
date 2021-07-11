import numpy as np
import matplotlib.pyplot as plt 
b = np.linspace(1000, 2000, 100)
a = np.linspace(1, 1000 ,100)
t1 = np.power(a, 3) + np.power(b, 3)
t2 = np.sqrt(a*b)*(a**2+b**2)
plt.plot(a, t1, label = r'$a^3+b^3$')
plt.plot(a, t2, '--', label = r'$\sqrt{ab}(a^2+b^2)$')
plt.legend()
# plt.semilogx()
plt.title('a < b')
plt.savefig('figure.png')
plt.show()