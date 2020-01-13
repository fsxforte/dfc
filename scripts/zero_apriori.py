import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import cm


sns.set(style="darkgrid")

correlations = [0, 0.25, 0.5, 0.75, 1]

def f(x, y, corr):
	'''
	Compute total margin.
	'''
	return x + y*(1-corr)

x = np.linspace(0, 1000, 100)
y = np.linspace(0, 1000, 100)

C = 100
PI = 400

C_margin = C * x
PI_margin = PI * y

X, Y = np.meshgrid(C_margin, PI_margin)

fig = plt.figure()
ax = plt.axes(projection='3d')

for correlation in correlations:
	Z = f(X, Y, correlation)
	ax.contour3D(X, Y, Z, 100, cmap=cm.coolwarm)

ax.set_xlabel('Collateral asset margin', linespacing = 10)
ax.set_ylabel('Reserve asset margin', linespacing = 10)
ax.set_zlabel('Total margin', linespacing = 10)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('Total margin with correlation', fontsize = 14)
ax.dit = 10
fig.savefig('../5d8dd7887374be0001c94b71/images/total_margin.png', bbox_inches = 'tight', dpi = 600)

plt.show()