"""
Total-energy conservation check for one production run -- rules out numerical
integration drift as the source of any observed lattice-distortion growth.
Reads `<prefix>/mem_data.hdf`'s saved energy components and saves
`<prefix>_energy_conservation.png`.
"""
import sys
import _pathsetup  # noqa: F401
import h5py
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <run_prefix>")
PREFIX = sys.argv[1]
H5PATH = f"{PREFIX}/mem_data.hdf"

with h5py.File(H5PATH, "r") as f:
    t = np.array(f["time/data"]).flatten()
    epot = np.array(f["Epot_ave/data"]).flatten()
    ekin = np.array(f["Ekin_ave/data"]).flatten()
    etot = np.array(f["Etot_ave/data"]).flatten()

drift_abs = etot - etot[0]
drift_rel = drift_abs / abs(etot[0]) if etot[0] != 0 else drift_abs
epot_drift = epot - epot[0]   # each component's OWN drift from t=0 -- E_pot/E_kin individually are O(1) Ha
ekin_drift = ekin - ekin[0]   # (bulk lattice/electronic energy), so a ~1e-5 Ha jump in E_tot is invisible
                              # against that background (see the top panel) unless plotted this way.

# Sanity check: does the saved Etot_ave actually equal Epot_ave + Ekin_ave at every step? If not, and if
# the residual itself carries the same staircase as Etot's drift, the jump is a bookkeeping artifact in
# how Etot is tracked/updated -- separate from the (apparently smooth) q/p dynamics Epot/Ekin reflect.
residual = etot - (epot + ekin)

print(f"=== {PREFIX} ===")
print(f"max|Etot - (Epot+Ekin)| over the run = {np.max(np.abs(residual)):.3e} Ha "
      f"(at t={t[np.argmax(np.abs(residual))]:.1f} a.u.) -- should be ~0 if Etot is literally their sum")
print(f"E_tot(t=0)   = {etot[0]:.8f} Ha")
print(f"E_tot(t=end) = {etot[-1]:.8f} Ha")
print(f"Absolute drift at t=end = {drift_abs[-1]:.3e} Ha")
print(f"Relative drift at t=end = {drift_rel[-1]:.3e}")
print(f"Max |drift| over the whole run = {np.max(np.abs(drift_abs)):.3e} Ha "
      f"(at t={t[np.argmax(np.abs(drift_abs))]:.1f} a.u.)")
print(f"E_pot drift at t=end = {epot_drift[-1]:.3e} Ha, E_kin drift at t=end = {ekin_drift[-1]:.3e} Ha "
      f"(should sum to the E_tot drift above -- whichever one accounts for most of it is where to look)")

fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
axes[0].plot(t, epot, label="E_pot", color="tab:orange")
axes[0].plot(t, ekin, label="E_kin", color="tab:green")
axes[0].plot(t, etot, label="E_tot", color="black", linewidth=2)
axes[0].set_ylabel("Energy (Ha)")
axes[0].legend()
axes[0].set_title(f"{PREFIX}: energy components vs. time")

axes[1].plot(t, epot_drift, label="E_pot(t) - E_pot(0)", color="tab:orange")
axes[1].plot(t, ekin_drift, label="E_kin(t) - E_kin(0)", color="tab:green")
axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[1].set_ylabel("Component drift (Ha)")
axes[1].legend()
axes[1].set_title("E_pot / E_kin drift, individually")

axes[2].plot(t, drift_abs, color="black")
axes[2].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[2].set_xlabel("time (a.u.)")
axes[2].set_ylabel("E_tot(t) - E_tot(0)  (Ha)")
axes[2].set_title("Total energy drift")

axes[3].plot(t, residual, color="tab:red")
axes[3].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[3].set_xlabel("time (a.u.)")
axes[3].set_ylabel("E_tot - (E_pot+E_kin)  (Ha)")
axes[3].set_title("Bookkeeping residual -- if this carries the same staircase as the panel above, "
                   "the jump is in how Etot is tracked, not in the real q/p dynamics")

plt.tight_layout()
plt.savefig(f"{PREFIX}_energy_conservation.png", dpi=150)
plt.show()
print(f"Saved {PREFIX}_energy_conservation.png")

# ---------------- zoomed view around the single largest step-to-step jump in Etot ----------------
jump_step = int(np.argmax(np.abs(np.diff(etot))))   # index i such that etot[i+1]-etot[i] is the largest jump
half_window = 15
lo, hi = max(0, jump_step - half_window), min(len(t), jump_step + half_window + 2)
print(f"\nLargest single-step Etot jump: index {jump_step} -> {jump_step + 1}, "
      f"t={t[jump_step]:.3f} -> {t[jump_step + 1]:.3f} a.u.")
print(f"  E_pot: {epot[jump_step]:.10f} -> {epot[jump_step + 1]:.10f}  (delta={epot[jump_step + 1] - epot[jump_step]:.3e})")
print(f"  E_kin: {ekin[jump_step]:.10f} -> {ekin[jump_step + 1]:.10f}  (delta={ekin[jump_step + 1] - ekin[jump_step]:.3e})")
print(f"  E_tot: {etot[jump_step]:.10f} -> {etot[jump_step + 1]:.10f}  (delta={etot[jump_step + 1] - etot[jump_step]:.3e})")

fig2, axes2 = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
idxs_zoom = np.arange(lo, hi)
axes2[0].plot(idxs_zoom, epot[lo:hi], "o-", color="tab:orange")
axes2[0].axvline(jump_step + 0.5, color="red", linestyle=":")
axes2[0].set_ylabel("E_pot (Ha)")
axes2[1].plot(idxs_zoom, ekin[lo:hi], "o-", color="tab:green")
axes2[1].axvline(jump_step + 0.5, color="red", linestyle=":")
axes2[1].set_ylabel("E_kin (Ha)")
axes2[2].plot(idxs_zoom, etot[lo:hi], "o-", color="black")
axes2[2].axvline(jump_step + 0.5, color="red", linestyle=":")
axes2[2].set_ylabel("E_tot (Ha)")
axes2[2].set_xlabel("saved step index")
fig2.suptitle(f"{PREFIX}: zoomed view around the largest single-step jump (index {jump_step})")
plt.tight_layout()
plt.savefig(f"{PREFIX}_jump_zoom.png", dpi=150)
plt.show()
print(f"Saved {PREFIX}_jump_zoom.png")
