#!/usr/local/bin/python3

# This script saves a 1D row/column of the Mach Number
# from a 3D Athena simulation ran with MPI. Choose two
# of x, y, or z coordinates in the overall domain for
# corresponding to the line of cells you wish to save.
# The script will save the Mach number for each snapshot
# in the specified directory.

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
    "-d", "--data",
    help="Required - Path to directory containing processor directories.",
    type=str,
    required=True
)
ap.add_argument(
    "--athinput",
    help="Path to athinput file.",
    type=str,
    required=True
)
ap.add_argument(
    "--out",
    help="Path to directory for saving concatenated files.",
    type=str,
    default="./",
    required=False
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
    help="Snapshot number (integer) at which to stop - exclusive. Default = 1", 
    type=int,
    required=False,
    default=1
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
#--------------------------------------------------------------------------
# organize data for opening
#--------------------------------------------------------------------------

saveloc = args.out
# create save destination
if not args.out == ".'":
    try:
        print(f'creating directory: {saveloc}')
        os.makedirs(saveloc, exist_ok=False)
    except FileExistsError:
        print(f'\'{saveloc}\' already exists!')


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
    datashape = (Nsnaps, Nz)
    print("Saving a z column")
if coords[1] == None: # y column if z,x chosen
    # zproc_slice = zcoord//nzloc
    # xproc_slice = xcoord//nxloc
    coreslice   = np.s_[zcoord//nzloc,:,xcoord//nxloc]
    local_slice = np.s_[zcoord%nzloc,:,xcoord%nxloc]
    datashape = (Nsnaps, Ny)
    print("Saving a y column")
if coords[2] == None: # x column if z,y chosen
    # zproc_slice = zcoord//nzloc
    # yproc_slice = ycoord//nyloc
    coreslice   = np.s_[zcoord//nzloc,ycoord//nyloc,:]
    local_slice = np.s_[zcoord%nzloc,ycoord%nyloc,:]
    datashape = (Nsnaps, Nx)
    print("Saving an x column")

# Construct 3D array of core numbers 
cores_array = np.arange(0,Ncores).reshape((Nzcores, Nycores, Nxcores))
# Create 2D array of the cores containing the desired slice
corelist = cores_array[coreslice]


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
rho_all = np.zeros(datashape)
rux_all = np.zeros(datashape)
ruy_all = np.zeros(datashape)
ruz_all = np.zeros(datashape)
eng_all = np.zeros(datashape)

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
        print("Opening :", filename)
        
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

        nx, ny, nz, nvar, nscalars= np.fromfile(file,dtype=np.int32,count=5)
        selfgrav_boolean, particles_boolean = np.fromfile(file,dtype=np.int32,count=2)

        gamma1, cs = np.fromfile(file, dtype=np.float64, count=2)
        t,dt = np.fromfile(file, dtype=np.float64, count=2)
        t_all[isnap] = t * UNIT_VALUES["unit_time"]

        x = np.fromfile(file, dtype=np.float64, count=nx)
        y = np.fromfile(file, dtype=np.float64, count=ny)
        z = np.fromfile(file, dtype=np.float64, count=nz)

        # x[0] is local, but x0 is global, etc.
        if j == 0:
            x0=x[0]
            y0=y[0]
            z0=z[0]
            dx=x[1]-x[0]
            dy=y[1]-y[0]
            dz=z[1]-z[0]

            # initiate the full-length x,y,z arrays
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

if zcoord == None:
    uz = ruz_all / rho_all
    mach_z = uz / UNIT_VALUES["unit_cs"]
    z_units = z_all * UNIT_VALUES["unit_length"]
    outpath = os.path.join(saveloc, f'mach_number_z.npz')
    print(f"Saving Mach number to {outpath}")
    np.savez(outpath, mach_z=mach_z, time=t_all, z=z_units)
if ycoord == None:
    uy = ruy_all / rho_all
    mach_y = uy / UNIT_VALUES["unit_cs"]
    y_units = y_all * UNIT_VALUES["unit_length"]
    outpath = os.path.join(saveloc, f'mach_number_y.npz')
    print(f"Saving Mach number to {outpath}")
    np.savez(outpath, mach_y=mach_y, time=t_all, y=y_units)
if xcoord == None:
    ux = rux_all / rho_all
    mach_x = ux / UNIT_VALUES["unit_cs"]
    x_units = x_all * UNIT_VALUES["unit_length"]
    outpath = os.path.join(saveloc, f'mach_number_x.npz')
    print(f"Saving Mach number to {outpath}")
    np.savez(outpath, mach_x=mach_x, time=t_all, x=x_units)
