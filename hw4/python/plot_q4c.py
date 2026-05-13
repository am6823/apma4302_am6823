import numpy as np
import matplotlib.pyplot as plt

N = np.array([16, 32, 64, 128])
Nu = np.array([1.70955360, 1.74000624, 1.74829694, 1.75041783])

Nu_benchmark = 4.884

plt.figure()
plt.plot(N, Nu, "o-", label="Firedrake result")
plt.axhline(Nu_benchmark, linestyle="--", label="Blankenbach benchmark")

plt.xlabel("Mesh resolution N")
plt.ylabel("Nusselt number Nu")
plt.title(r"Nusselt number vs mesh size for $Ra=10^4$")
plt.xticks(N)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("result/q4c_Nu_vs_mesh.png", dpi=300)
plt.show()