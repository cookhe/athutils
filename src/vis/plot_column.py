#!/usr/local/bin/python3

# This script plots a 1D row/column from a 3D Athena simulation ran
# with MPI. Choose two of x, y, or z coordinates in the 
# overall domain for the line of cells.

# Author: Harrison Cook


import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import add_slash_if_none, read_units
from ReadAthena import parse_dimensions, parse_athinput



# parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "--data",
    help="Required - Path to directory containing processor directories.",
    type=str,
    required=True
)
ap.add_argument(
    "--out",
    help="Path to directory for saving concatenated files.",
    type=str,
    default="column_plot"
)
ap.add_argument(
    "--athinput",
    help="Path to athinput file.",
    type=str,
    required=True
)
ap.add_argument(
    "--start",
    help="Snapshot number (integter) at which to start. Default = 0",
    type=int,
    required=False,
    default=0
)
ap.add_argument(
    "--stop",
    help="Snapshot number (integer) at which to stop - exclusive. Continues to end if option not used.",
    type=int,
    required=False
)
ap.add_argument(
    "--xcoord",
    help="Coordinate along x-direction to plot. Only one of -x, -y, and -z can be set.",
    type=int,
    required=False,
    default=None
)
ap.add_argument(
    "--ycoord",
    help="Coordinate along y-direction to plot. Only one of -x, -y, and -z can be set.",
    type=int,
    required=False,
    default=None
)
ap.add_argument(
    "--zcoord",
    help="Coordinate along z-direction to plot. Only one of -x, -y, and -z can be set.",
    type=int,
    required=False,
    default=None
)
ap.add_argument(
    "--show",
    help="Show the plot instead of save. Default False.",
    required=False,
    action="store_true"
)

# args = vars(ap.parse_args())
args = ap.parse_args()
start = args.start
stop = args.stop
Nsnaps = stop - start

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
datadir = add_slash_if_none(args.data)
jobname = os.listdir(os.path.join(datadir,'id0'))[0].split('.')[0]

#--------------------------------------------------------------------------
# input file parser to get grid information
#--------------------------------------------------------------------------
ATHINPUT_VALUES = parse_athinput(args.athinput)
Nz = ATHINPUT_VALUES["Nx3"]
Ny = ATHINPUT_VALUES["Nx2"]
Nx = ATHINPUT_VALUES["Nx1"]

Nzcores = ATHINPUT_VALUES["NGrid_x3"]
Nycores = ATHINPUT_VALUES["NGrid_x2"]
Nxcores = ATHINPUT_VALUES["NGrid_x1"]

Ncores = Nzcores * Nycores * Nxcores
nzloc = Nz//Nzcores
nyloc = Ny//Nycores
nxloc = Nx//Nxcores

#--------------------------------------------------------------------------
# units
#--------------------------------------------------------------------------
UNIT_VALUES = read_units()
unit_density = UNIT_VALUES["unit_density"] # g cm^-3
unit_length = UNIT_VALUES["unit_length"] # cm
H_cm = unit_length
H_au = H_cm /  1.495979e+13
#--------------------------------------------------------------------------
# organize data for opening
#--------------------------------------------------------------------------


# create save destination
if not args.show:
    newdir = add_slash_if_none(args.out)
    try:
        print(f'creating directory: {newdir}')
        os.makedirs(newdir, exist_ok=False)
    except FileExistsError:
        print(f'\'{newdir}\' already exists!')


filename_root = os.getcwd().split('/')[-1]

# Collect coordinates
zcoord = args.zcoord
ycoord = args.ycoord
xcoord = args.xcoord
coords = np.array([zcoord, ycoord, xcoord])

# Check that at least and only one of the coordinate values is an integer
if np.sum(coords != None) > 2:
    raise ValueError(f"May set only two of zcoord, ycoord, or xcoord. Values are: {zcoord, ycoord, xcoord}")
elif np.sum(coords != None) < 2:
    raise ValueError(f"Must set two of zcoord, ycoord, or xcoord can be non-zero. Values are: {zcoord, ycoord, xcoord}")


# Get the slice of cores and define local coordinate
if coords[0] == None: # z column if x,y chosen
    # yproc_slice = ycoord//nyloc
    # xproc_slice = xcoord//nxloc
    coreslice   = np.s_[:,ycoord//nyloc,xcoord//nxloc]
    local_slice = np.s_[:,ycoord%nyloc,xcoord%nxloc]
    plotshape = (Nsnaps, Nz)
if coords[1] == None: # y column if z,x chosen
    # zproc_slice = zcoord//nzloc
    # xproc_slice = xcoord//nxloc
    coreslice   = np.s_[zcoord//nzloc,:,xcoord//nxloc]
    local_slice = np.s_[zcoord%nzloc,:,xcoord%nxloc]
    plotshape = (Nsnaps, Ny)
if coords[2] == None: # x column if z,y chosen
    # zproc_slice = zcoord//nzloc
    # yproc_slice = ycoord//nyloc
    coreslice   = np.s_[zcoord//nzloc,ycoord//nyloc,:]
    local_slice = np.s_[zcoord%nzloc,ycoord%nyloc,:]
    plotshape = (Nsnaps, Nx)

# Construct 3D array of core numbers 
cores_array = np.arange(0,Ncores).reshape((Nzcores, Nycores, Nxcores))
# Create 2D array of the cores containing the desired slice
corelist = cores_array[coreslice]

print("plotshape:", plotshape)

###################################################################
def getlevels(array, resolution, log10=True):
    if log10 == True:
        arraymin = np.nanmin(np.log10(array))
        arraymax = np.nanmax(np.log10(array))
        return np.linspace(arraymin, arraymax, resolution)
    else:
        arraymin = np.nanmin(array)
        arraymax = np.nanmax(array)
        return np.linspace(arraymin, arraymax, resolution)
###################################################################
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Create empty containers for the number of snapshots and image size
rho_all = np.zeros(plotshape)
rux_all = np.zeros(plotshape)
ruy_all = np.zeros(plotshape)
ruz_all = np.zeros(plotshape)
eng_all = np.zeros(plotshape)

t_all   = np.zeros(Nsnaps)
dt_all  = np.zeros(Nsnaps)

# Loop through the number of desired snapshots
for isnap, snapnum in enumerate(range(start, stop)):


    for j, jcore in enumerate(corelist.flatten()):
        # Read binary files
        # filename = binlist[jcore][isnap]
        if jcore == 0:
            # filename matches structure: ./output/id0/<jobname>.<snapshot>.bin
            filename = os.path.join(datadir, f"id{jcore}",f"{jobname}.{snapnum:0>4}.bin")
        else:
            # filename matches structure: ./output/id<corenumber>/<jobname>-.<snapshot>.bin
            filename = os.path.join(datadir, f"id{jcore}",f"{jobname}-id{jcore}.{snapnum:0>4}.bin")
        print(filename)
        
        # Confirm it's a C binary
        try:
            file = open(filename, 'rb')
        except:
            print('data must be <binary_dump>')
            raise SystemExit

        # # read data from binary file
        file.seek(0,2)
        eof = file.tell()
        file.seek(0,0)

        coordsys = np.fromfile(file, dtype=np.int32, count=1)[0]
        #print("coordsys=",coordsys)

        nx, ny, nz, nvar, nscalars= np.fromfile(file,dtype=np.int32,count=5)
        selfgrav_boolean, particles_boolean = np.fromfile(file,dtype=np.int32,count=2)
        # print("nx,ny,nz,nvar,nscalars,selfgrav_boolean,particles_boolean=",
        # nx,ny,nz,nvar,nscalars,selfgrav_boolean,particles_boolean)
        # print("file, nx, ny, nz", filename, nx, ny, nz)
        # continue

        gamma1, cs = np.fromfile(file, dtype=np.float64, count=2)
        print("gamma1,cs=", gamma1, cs)
        break
        t,dt = np.fromfile(file, dtype=np.float64, count=2)
        t_all[isnap] = t * ATHINPUT_VALUES["unit_time"]
        # print(t)
        # sys.exit()

        x = np.fromfile(file, dtype=np.float64, count=nx)
        y = np.fromfile(file, dtype=np.float64, count=ny)
        z = np.fromfile(file, dtype=np.float64, count=nz)

        # x[0] is local, but x0 is global
        if j == 0:
            x0=x[0]
            y0=y[0]
            z0=z[0]
            dx=x[1]-x[0]
            dy=y[1]-y[0]
            dz=z[1]-z[0]

            # initiate the full-length x,y,z arrays
            print("PP")
            x_all = x
            y_all = y
            z_all = z

        # add next set of coordinates if first value follows existing (limiting sum to machine precision)
        if np.isclose(x[0], x_all[-1] + dx): # np.isclose() used to handle machine precision deviations.
            x_all = np.concatenate((x_all,x), axis=0)

        if np.isclose(y[0], y_all[-1] + dy):
            y_all = np.concatenate((y_all,y), axis=0)

        if np.isclose(z[0], z_all[-1] + dz):
            z_all = np.concatenate((z_all,z), axis=0)

        # recurrence relations for each axis used to index monolithical dataset
        ix0 = np.int32(np.round((x[0]-x0)/dx, 12))
        ix1 = ix0 + nx
        iy0 = np.int32(np.round((y[0]-y0)/dy, 12))
        iy1 = iy0 + ny
        iz0 = np.int32(np.round((z[0]-z0)/dz)) # removed the 12 order, becsuse rounded improperly
        iz1 = iz0 + nz

        localshape = (nz, ny, nx)
        count = np.prod(localshape) #nx*ny*nz

        rho = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
        rux = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
        ruy = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
        ruz = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
        eng = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]

        #print(rho.shape)
        #print('core=',jcore,'rhominmax=',rho[0,...].min(),rho[0,...].max()," \n")

        if file.tell() != eof:
            print('Error: Too few bytes read.')

        file.close()

        # Location in full snapshot
        if zcoord == None:
            coreloc = np.s_[iz0:iz1]
        if ycoord == None:
            coreloc = np.s_[iy0:iy1]
        if xcoord == None:
            coreloc = np.s_[ix0:ix1]

        # now populate the *_all arrays to recreate the monolithical dataset
        rho_all[isnap][coreloc] = rho * UNIT_VALUES["unit_density"]
        rux_all[isnap][coreloc] = rux * UNIT_VALUES["unit_density"] * UNIT_VALUES["unit_velocity"]
        ruy_all[isnap][coreloc] = ruy * UNIT_VALUES["unit_density"] * UNIT_VALUES["unit_velocity"]
        ruz_all[isnap][coreloc] = ruz * UNIT_VALUES["unit_density"] * UNIT_VALUES["unit_velocity"]
        eng_all[isnap][coreloc] = eng * UNIT_VALUES["unit_edens"]

        # print(jcore,iz0,iz1,iy0,iy1,ix0,ix1)
# sys.exit()

uz = ruz_all / rho_all
uy = ruy_all / rho_all
ux = rux_all / rho_all
mach_z = np.abs(uz) / UNIT_VALUES["unit_cs"]
mach_y = np.abs(uy) / UNIT_VALUES["unit_cs"]
mach_x = np.abs(ux) / UNIT_VALUES["unit_cs"]


xlim_min = -5
xlim_max =  5



max_mach_z = np.max(mach_z)
max_mach_y = np.max(mach_y)
max_mach_x = np.max(mach_x)
# print(t)


plotslice = np.s_[:,:] # no slice
# plotslice = np.s_[890:950,420:540] #  roe 45 days
# plotslice = np.s_[1080:1160,340:620] # roe final shock
# plotslice = np.s_[1080:1190,340:620] # roe h-correct final shock
# plotslice = np.s_[910:970,420:540] # exact 45 days
# plotslice = np.s_[1150:1270,280:680] # hllc final shock (snapshot 0028)
# plotslice = np.s_[910:970,420:540] # hllc 45 days
# plotslice = np.s_[1175:1265,340:620] # hlle final shock (snapshot 0029)
# plotslice = np.s_[890:950,420:540] # hllc 45 days

x_units = x_all * unit_length
y_units = y_all * unit_length
z_units = z_all * unit_length

ylim_min = 1e-2
ylim_max_factor = 1.1

for isnap, snapnum in enumerate(range(start,stop)):
    time_days = t_all[isnap]/86400
    print(isnap, time_days, "d")

    plt.figure()
    ax = plt.gca()
    
    ax.set_title(f't = {time_days:.0} days')

    if zcoord == None:
        # ax.plot(z_units/H_cm, max_mach_z[isnap])
        ax.plot(z_units/H_cm, rho_all[isnap])
    if ycoord == None:
        # ax.plot(y_units/H_cm, max_mach_y[isnap])
        ax.plot(y_units/H_cm, rho_all[isnap])
    if xcoord == None:
        # ax.plot(x_units/H_cm, max_mach_x[isnap])
        ax.plot(x_units/H_cm, rho_all[isnap])

    # Plot Mach = 1 line
    # ax.axhline(1, color='Grey', linestyle='--', label=r"$Ma = 1$")

    if zcoord == None:
        # ax.set(xlabel='z/H', ylabel=r'$|v_z|\ /\ c_s$')
        # ax.set_ylim(ylim_min, ylim_max_factor * max_mach_z)

        ax.set(xlabel='z/H', ylabel=r'Density (g/cm$^3$)')
        ax.set_ylim(ylim_min, ylim_max_factor * rho_all[isnap,:,ycoord,xcoord])
        # ax.set_ylim(ylim_min, 100) # use for log scale zoom on disk
        # ax.set_ylim(0, 100) # use for linear scale zoom on disk

    if ycoord == None:
        # ax.set(xlabel='y/H', ylabel=r'$|v_y|\ /\ c_s$')
        # ax.set_ylim(ylim_min, ylim_max_factor * max_mach_y)

        ax.set(xlabel='z/H', ylabel=r'Density (g/cm$^3$)')
        ax.set_ylim(ylim_min, ylim_max_factor * rho_all[isnap,zcoord,:,xcoord])
        # ax.set_ylim(ylim_min, 100) # use for log scale zoom on disk
        # ax.set_ylim(0, 100) # use for linear scale zoom on disk

    if xcoord == None:
        # ax.set(xlabel='x/H', ylabel=r'$|v_x|\ /\ c_s$')
        # ax.set_ylim(ylim_min, ylim_max_factor * max_mach_x)

        ax.set(xlabel='z/H', ylabel=r'Density (g/cm$^3$)')
        ax.set_ylim(ylim_min, ylim_max_factor * rho_all[isnap,zcoord,ycoord,:])
        # ax.set_ylim(ylim_min, 100) # use for log scale zoom on disk
        # ax.set_ylim(0, 100) # use for linear scale zoom on disk


    ax.set_yscale('log')
    # ax.set_xlim(-5, 5)
    # ax.set_xticks(np.linspace(-5,5,11))
    ax.grid('on', ls=':')
    ax.legend(loc="upper left")
    
    # Pick one of these
    if args.show:
        plt.show()
    else:
        print('Saved image')
        plt.savefig(os.path.join(newdir, f'density_z_{snapnum:0>4}.png'), format='png', bbox_inches="tight", dpi=600)

    plt.close()

    