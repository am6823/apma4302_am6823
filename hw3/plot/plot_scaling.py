import matplotlib.pyplot as plt
import numpy as np

data = [
    (2, 1,  33,  2.1924e-02, 0.000809754, 2),
    (2, 2,  33,  2.4570e-02, 0.000809754, 2),
    (2, 4,  33,  7.2220e-03, 0.000809754, 2),
    (3, 1,  65,  4.8361e-02, 0.000202508, 2),
    (3, 2,  65,  2.5707e-02, 0.000202508, 2),
    (3, 4,  65,  3.7250e-02, 0.000202508, 2),
    (4, 1, 129,  2.4773e-01, 5.06487e-05, 2),
    (4, 2, 129,  1.3535e-01, 5.06487e-05, 2),
    (4, 4, 129,  1.3021e-01, 5.06487e-05, 2),
    (5, 1, 257,  1.3194e+00, 1.26657e-05, 2),
    (5, 2, 257,  5.9666e-01, 1.26657e-05, 2),
    (5, 4, 257,  5.0224e-01, 1.26657e-05, 2),
    (6, 1, 513,  8.3289e+00, 3.16693e-06, 2),
    (6, 2, 513,  2.5998e+00, 3.16693e-06, 2),
    (6, 4, 513,  2.1458e+00, 3.16693e-06, 2),
]

refines = sorted(set(d[0] for d in data))
nprocs  = sorted(set(d[1] for d in data))
grids   = {d[0]: d[2] for d in data}
colors  = {1: 'steelblue', 2: 'darkorange', 4: 'seagreen'}

times  = {p: [] for p in nprocs}
errors = {p: [] for p in nprocs}
sizes  = []

for r in refines:
    sizes.append(grids[r])
    for p in nprocs:
        row = next(d for d in data if d[0]==r and d[1]==p)
        times[p].append(row[3])
        errors[p].append(row[4])

sizes = np.array(sizes)

# Figure 1 : run time vs grid size 
fig, ax = plt.subplots(figsize=(6, 4))
for p in nprocs:
    ax.loglog(sizes, times[p], 'o-', color=colors[p],
              linewidth=2, markersize=6, label=f'{p} proc(s)')

ref_x = np.array([sizes[0], sizes[-1]])
ref_y = times[1][0] * (ref_x / sizes[0])**3
ax.loglog(ref_x, ref_y, 'k--', linewidth=1, label=r'$O(N^3)$ ref')

ax.set_xlabel('Grid size $N$ ($N \\times N$)', fontsize=12)
ax.set_ylabel('SNESSolve time (s)', fontsize=12)
ax.set_title('Run time vs grid size ($\\gamma=100$, $p=3$)', fontsize=11)
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('scaling_time.png', dpi=150)

# Figure 2 : relative error vs grid size 
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(sizes, errors[1], 'o-', color='steelblue', linewidth=2, markersize=6)

ref_y2 = errors[1][0] * (sizes[0] / sizes)**2
ax.loglog(sizes, ref_y2, 'k--', linewidth=1, label=r'$O(h^2)$ ref')

ax.set_xlabel('Grid size $N$ ($N \\times N$)', fontsize=12)
ax.set_ylabel(r'$\|u - u_{\mathrm{exact}}\|_2 / \|u_{\mathrm{exact}}\|_2$', fontsize=12)
ax.set_title(r'Relative error vs grid size ($\gamma=100$, $p=3$)', fontsize=11)
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('scaling_error.png', dpi=150)

plt.show()

# Print table
print(f"{'Grid':>8} {'NProc':>6} {'Time (s)':>10} {'Rel error':>12} {'Newton':>7}")
print("-" * 50)
for d in data:
    refine, nproc, grid, t, err, nit = d
    print(f"{grid:>5}x{grid:<3} {nproc:>6} {t:>10.4f} {err:>12.2e} {nit:>7}")