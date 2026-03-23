import matplotlib.pyplot as plt
import numpy as np

iterations = [0, 1, 2]
residuals  = [8.867115559165e-01, 7.777424323105e-05, 1.905055223210e-11]

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(iterations, residuals, 'o-', color='steelblue', linewidth=2, markersize=7)

ax.set_xlabel('Newton iteration', fontsize=12)
ax.set_ylabel(r'$\|F(u)\|_2$', fontsize=12)
ax.set_title(r'Nonlinear residual convergence ($\gamma=100$, $p=3$, $65\times65$ grid)', fontsize=11)
ax.set_xticks(iterations)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.axhline(1e-10, color='red', linestyle='--', linewidth=1, label=r'tolerance $10^{-10}$')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('convergence_gamma100.png', dpi=150)
plt.show()