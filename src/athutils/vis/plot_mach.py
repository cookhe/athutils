#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import scienceplots
# import plotting
from ReadAthena import parse_dimensions, parse_athinput
from functions import read_units

plt.style.use('science')
# plt.rcParams.update({
    # "legend.fontsize": 8,
    # "legend.title_fontsize": 8,
# })

# Choose how many disk sale heights to plot.
disk_thickness_H = 5
zfigname_base = f"output/max_mach/max_mach_z_{disk_thickness_H}H_compare_zooms"
tfigname_base = f"output/max_mach/max_mach_t_{disk_thickness_H}H_compare_zooms"

runRootDir = "/work2/07139/hecook/stampede3/supernova-runs/production"
machFileName  = "mach_number_z.npz"
unitsFileName = "unit_values.json"

def loadMachData(location, machFilename, unitsFilename, disk_thickness):
    dataPath = os.path.join(location, machFilename)
    unitsPath = os.path.join(location, unitsFilename)
    assert os.path.isfile(dataPath), f"File not found:" + dataPath
    assert os.path.isfile(unitsPath), f"File not found:" + unitsPath

    with np.load(dataPath) as mf:
        z = mf['z']
        t = mf['time']
        m = mf['mach_z']

    units = read_units(unitsPath)

    return z, t, m, units

def trimDataToDisk(data, z_data_H, disk_thickness_H):
    """Trim the data to the disk region between 0 the specified number of scale heights.

    Parameters
    ----------
    data : numpy.ndarray
        Data array with shape (time_steps, z_points).
    z_data_H : numpy.ndarray
        Z coordinates of the data.    
    disk_thickness_H : float
        Number of scale heights to consider for the disk region.
    """
    mask = np.argwhere((z_data_H >= 0) & (z_data_H <= disk_thickness_H))

    return z_data_H[mask], data[:, mask]


# Open files
locs = [
    "solver-test/hlle/",
    "midplane1e5/",
    "midplane1e5-zoom/output/analysis-L24-C16/",
    "midplane1e5-zoom/output/analysis-L12-C16/",
    "midplane1e5-zoom/output/analysis-L24-C16-s0p075/",
    "onezone/analysis/"
]

z = []
time = []
mach = []
unit = []
mach_z_max = []
mach_t_max = []

for li, loc in enumerate(locs):
    zi, ti, mi, ui = loadMachData(os.path.join(runRootDir, loc), machFileName, unitsFileName, disk_thickness_H)
    zi_trimmed, mi_trimmed = trimDataToDisk(mi, zi / ui["scale_height"], disk_thickness_H)
    
    mi_nonan = []
    ti_nonan = []
    # check for NaNs and remove
    if np.isnan(mi_trimmed).any():
        for ind, mi_test in enumerate(mi_trimmed):
            if not np.isnan(mi_test).any():
                mi_nonan.append(mi_test)
                ti_nonan.append(ti[ind])
        ti = np.array(ti_nonan)
        mi_trimmed = np.array(mi_nonan)
                

    ti_sort_ind = np.argsort(ti)
    ti = ti[ti_sort_ind]
    mi_trimmed = mi_trimmed[ti_sort_ind]
    
    m_z_max = np.max(mi_trimmed, axis=0)
    m_t_max = np.array([np.max(timestep) for timestep in mi_trimmed])
    
    # Add to the lists   
    z.append(zi_trimmed)
    time.append(ti)
    mach.append(mi_trimmed)
    unit.append(ui)
    mach_z_max.append(m_z_max)
    mach_t_max.append(m_t_max)
    # if li == 4:
        # raise SystemExit
    

labels = [
    r"$10^3$",
    r"$10^5$",
    r"$10^5$ L24",
    r"$10^5$ L12",
    r"$10^5$ L24 onezone",
    r"$10^5$ L24 $\sigma$=3dx"
]
# lines = ["-", "-.", ":", "--"]
# linecycler = cycle(lines)
clinspace = np.linspace(0, 1, len(mach_z_max))
# Vs z
plt.figure()
for j, (zj, max_mach_zj) in enumerate(zip(z, mach_z_max)):
    # plt.plot(zj, max_mach_zj, ls=next(linecycler), label=labels[j], color=plt.cm.cool(clinspace[j]))
    plt.plot(zj, max_mach_zj, label=labels[j], color=plt.cm.cool(clinspace[j]))
plt.gca().axhline(1, color="k", ls=":", label=r"$Ma=1$")
plt.yscale("log")
plt.xlabel(r"$z/H$")
plt.title("Maximum Mach Number By Zone")
plt.ylabel(r"$Ma$")
# plt.xlim(0, disk_thickness_H)
plt.ylim(1e-6,3e2)
plt.legend(title=r"SN Location ($R_{\rm s}$)", fontsize=6, title_fontsize=6, loc="lower left")

for figfmt in ("png", "pdf"):
    zfigname = f"{zfigname_base}.{figfmt}"
    plt.savefig(zfigname, format=figfmt, bbox_inches="tight", dpi=300)

plt.close()

# Vs time
# plt.figure()
# Use these lines to plot in days
# plt.xlabel("Time (days)")
# plt.plot(t_disk_1e3 / 86400, max_mach_t_1e3, marker='^', markersize=1.5, ls="-.", label=r"$10^3\,R_s$", color=plt.cm.cool(0.25))
# plt.plot(t_disk_1e5 / 86400, max_mach_t_1e5, marker='^', markersize=1.5, ls="--", label=r"$10^5\,R_s$", color=plt.cm.cool(0.5))
# plt.plot(t_disk_1e5z / 86400, max_mach_t_1e5z, marker='^', markersize=1.5, label=r"$10^5\,R_s$", color=plt.cm.cool(0.75))
# plt.plot(t_disk_1e5zz / 86400, max_mach_t_1e5zz, marker='^', markersize=1.5, label=r"$10^5\,R_s$", color=plt.cm.cool(1))

# Use these lines to plot in orbits
# plt.xlabel("Time (orbits)")
# plt.plot(t_disk_1e3 / Omega_1e3, max_mach_t_1e3, marker='^', markersize=1.5, ls="-.", label=r"$10^3\,R_s$", color=plt.cm.cool(0))
# plt.plot(t_disk_1e5 / Omega_1e5, max_mach_t_1e5, marker='^', markersize=1.5, label=r"$10^5\,R_s$", color=plt.cm.cool(0.25))
# plt.plot(t_disk_1e5z / Omega_1e5z, max_mach_t_1e5z, marker='^', markersize=1.5, ls="--", label=r"$10^5\,R_s$ L24", color=plt.cm.cool(0.5))
# plt.plot(t_disk_1e5zz / Omega_1e5z, max_mach_t_1e5zz, marker='^', markersize=1.5, ls="--", label=r"$10^5\,R_s$ L12", color=plt.cm.cool(0.75))
# plt.plot(t_disk_1e5one / Omega_1e5one, max_mach_t_1e5one, marker='^', markersize=1.5, ls="--", label=r"$10^5\,R_s$ L12 onezone", color=plt.cm.cool(0.85))
# plt.plot(t_disk_1e5s / Omega_1e5s, max_mach_t_1e5s, marker='^', markersize=1.5, ls="--", label=r"$10^5\,R_s$ L12 $\sigma$=3dx", color=plt.cm.cool(1))

# plt.gca().axhline(1, color="k", ls=":", label=r"$Ma=1$")

# plt.yscale("log")
# plt.xscale("log")
# plt.title("Maximum Mach Number Timeseries")
# plt.ylabel(f"max$[Ma(z \leq {disk_thickness_H}H)]$")
# plt.legend(fontsize=8, title_fontsize=8, location="upper right")

# for figfmt in ("png", "pdf"):
    # tfigname = f"{tfigname_base}.{figfmt}"
    # plt.savefig(tfigname, format=figfmt, bbox_inches="tight", dpi=300)

# plt.close()

