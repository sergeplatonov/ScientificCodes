import numpy as np
from types import SimpleNamespace
from matplotlib import pyplot
np.set_printoptions(precision=4, linewidth=200)

def SetUpH(p, doPlot = False):
	# Simple tight binding model for a 1D chain with an onsite potential
	def pot(x):
		pvec = np.zeros(p.L, dtype=float)
		for cnt, xval in enumerate(x):
			pvec[cnt] = 2*p.t \
				+ p.px2*np.power((xval - p.L/2),2) \
				+ p.px4*np.power((xval - p.L/2),4)
		return pvec
	if doPlot: pyplot.plot(list(range(p.L)), pot(list(range(p.L))))
	
	H = np.diag(pot(list(range(p.L))))
	
	for cnt in range(p.L-1):
		H[cnt,cnt+1] = -p.t
		H[cnt+1,cnt] = -p.t
		
	return H

# Parameters for the Harmonic H
# L ist the number of sites
# px2 is the prefactor of quadratic potential
# px4 is the prefactor of the x^4 potential
# t is the hopping
px2set = 0.0000004
p  = SimpleNamespace(L = 400, px2 = px2set, px4 = 0.00*px2set, t=1.0)
# Parameters for the Anharmonic H.
p4 = SimpleNamespace(L = 400, px2 = px2set, px4 = 0.03*px2set, t=1.0)

# Harmonic H
ham = SetUpH(p, False)
# Anharmonic H
ham4 = SetUpH(p4, False)
#pyplot.draw()

# Compute eigensystems
evals, evecs = np.linalg.eigh(ham)
evals4, evecs4 = np.linalg.eigh(ham4)

# Numbers of eigenstates to plot
n = range(7)

# Print eigenenergies
print("Eigenenergies harmonic potential"); 
print(evals[n])
print("Eigenenergies harmonic potential, normalized to 2 eval[0]"); 
print(evals[n]/(2*evals[0]))
print("Eigenenergies anharmonic potential"); 
print(evals4[n])
print("Eigenenergies anharmonic potential, normalized to 2 eval4[0]"); 
print(evals4[n]/(2*evals4[0]))
print("Difference of the normalized anharmonic eigenvalues"); 
evalsdiff = np.zeros(len(n)-1)
for cnt in range(len(n)-1):
	evalsdiff[cnt] = \
		(evals4[cnt + 1]/(2*evals4[0])) - \
		(evals4[cnt]/(2*evals4[0]))
print(evalsdiff)

# Plot wavefunctions
f, axarr = pyplot.subplots(len(n), sharex=True)
for cntn in n:
	axarr[cntn].plot(range(p.L), np.abs(evecs[:, cntn])**2)
	axarr[cntn].plot(range(p.L), np.abs(evecs4[:, cntn])**2)

axarr[0].set_title("|Psi|^2 for V = x^2 (blue) and V = x^2 + a x^4, a = 0.03 (green)")
pyplot.show()
