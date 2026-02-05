"""Functions to read Athena files."""

import os
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional, List, Dict

ATHINPUT_TYPES = {
    "problem_id"   : str,
    "num_domains"  : int,

    "cour_no"      : float,
    "nlim"         : int,
    "tlim"         : float,
    "level"        : int,    # identifies the Static mesh refinement level (0=root)
    "Nx1"          : int,    # Number of zones in X-direction
    "x1min"        : float,  # minimum value of X
    "x1max"        : float,  # maximum value of X
    "bc_ix1"       : int,    # inner-I (X) boundary flag (periodic)
    "bc_ox1"       : int,    # outer-I (X) boundary flag (periodic)
    "NGrid_x1"     : int,    # dimension divided into NGrid MPI blocks

    "Nx2"          : int,     # Number of zones in X2-direction
    "x2min"        : float,   # minimum value of X2
    "x2max"        : float,   # maximum value of X2
    "bc_ix2"       : int,     # inner-J (Y) boundary flag (periodic)
    "bc_ox2"       : int,     # outer-J (Y) boundary flag (periodic)
    "NGrid_x2"     : int,     # dimension divided into NGrid MPI blocks

    "Nx3"          : int,     # Number of zones in X3-direction
    "x3min"        : float,   # minimum value of X3
    "x3max"        : float,   # maximum value of X3
    "bc_ix3"       : int,     # inner-K (Z) boundary flag (outflow)
    "bc_ox3"       : int,     # outer-K (Z) boundary flag (outflow)
    "NGrid_x3"     : int,     # dimension divided into NGrid MPI blocks

    "rho0"         : float,   # disk midplane density; code units
    "d_floor"      : float,   # density floor
    "snEng"        : float,   # sn energy; code units (default = 0, no SN)
    "snMass"       : float,   # 10 M_sun - mass of sn ejecta in code units (default=0, no SN)
    "epsilon"      : float,   # fraction of SN energy in kinetic energy
    "sigma"        : float,   # radius of SN in units of domain length: sigma =  3 x (x1max-x1min)/Nx1
    "sn1"          : float,
    "sn2"          : float,
    "sn3"          : float,
    "omega"        : float,   # angular velocity of rotation
    "gamma"        : float,   # adiabatic index (radiation dominated)
    "adi_csound"   : float,   # adiabatic sound speed
    "shear"        : float,   # Logarithmic shear rate
}

SKIP_KEYS = [
    "problem",
    "author",
    "config",
    "problem_id",
    "maxout",
    "out_fmt",
    "out",
    "dt",
]

@dataclass
class AthenaSlice:
    coords: dict       # {"x": xarr, "y": yarr, "z": zarr}
    data: dict
    t: float
    dt: float
    snapshot: int
    axis: str
    index: int

    def save_npz(self, filename):
        """Save slice to compressed NPZ format."""
        np.savez_compressed(
            filename,
            t=self.t,
            dt=self.dt,
            snapshot=self.snapshot,
            axis=self.axis,
            index=self.index,
            **self.coords,
            **self.data
        )

    def save_hdf5(self, filename, mode='a'):
        """Save slice to HDF5 format.
        
        Parameters
        ----------
        filename : str
            HDF5 file path
        mode : str
            File mode ('w' for write, 'a' for append)
        """
        with h5py.File(filename, mode) as f:
            # Create group for this snapshot
            grp_name = f"snapshot_{self.snapshot:04d}"
            
            # Remove existing group if present
            if grp_name in f:
                del f[grp_name]
            
            grp = f.create_group(grp_name)
            
            # Save metadata
            grp.attrs['t'] = self.t
            grp.attrs['dt'] = self.dt
            grp.attrs['snapshot'] = self.snapshot
            grp.attrs['axis'] = self.axis
            grp.attrs['index'] = self.index
            
            # Save coordinates
            coords_grp = grp.create_group('coords')
            for key, val in self.coords.items():
                coords_grp.create_dataset(key, data=val, compression='gzip')
            
            # Save data fields
            data_grp = grp.create_group('data')
            for key, val in self.data.items():
                data_grp.create_dataset(key, data=val, compression='gzip')

    @classmethod
    def load_hdf5(cls, filename, snapshot):
        """Load a slice from HDF5 file.
        
        Parameters
        ----------
        filename : str
            HDF5 file path
        snapshot : int
            Snapshot number to load
            
        Returns
        -------
        AthenaSlice
        """
        with h5py.File(filename, 'r') as f:
            grp_name = f"snapshot_{snapshot:04d}"
            if grp_name not in f:
                raise KeyError(f"Snapshot {snapshot} not found in {filename}")
            
            grp = f[grp_name]
            
            # Load metadata
            t = grp.attrs['t']
            dt = grp.attrs['dt']
            snap = grp.attrs['snapshot']
            axis = grp.attrs['axis']
            index = grp.attrs['index']
            
            # Load coordinates
            coords = {key: val[:] for key, val in grp['coords'].items()}
            
            # Load data
            data = {key: val[:] for key, val in grp['data'].items()}
            
            return cls(coords=coords, data=data, t=t, dt=dt, 
                      snapshot=snap, axis=axis, index=index)

class Athena:
    def __init__(self, file_athinput, datadir):
        self.inputs = self.parse_athinput(file_athinput)
        self.problem_id = self.inputs["problem_id"]
        self.Ncores = self.inputs["Ncores"]
        self.datadir = datadir
        self.AXES = {
            "x3": {"coord": 0, "plane": {"x2", "x1"}},
            "x2": {"coord": 1, "plane": {"x3", "x1"}},
            "x1": {"coord": 2, "plane": {"x3", "x2"}}
        }

    
    def slice(self, snapshot, *, x3=None, x2=None, x1=None, fields=None):
        """Extract a 2D slice from a 3D snapshot.
        
        Parameters
        ----------
        snapshot : int
            Snapshot number
        x3, x2, x1 : int, optional
            Index along the axis to slice (exactly one must be specified)
        fields : list, optional
            List of fields to extract. If None, extracts all standard fields.
            
        Returns
        -------
        AthenaSlice
        """
        if fields is None:
            fields = ["rho", "rux1", "rux2", "rux3", "eng"]

        return self._construct_2Dslice(
            snapshot=snapshot,
            x3index=x3,
            x2index=x2,
            x1index=x1,
            fields=fields
        )
    
    def slice_to_hdf5(self, snapshots, hdf5_file, *, x3=None, x2=None, x1=None, fields=None):
        """Extract slices from multiple snapshots and save to a single HDF5 file.
        
        Parameters
        ----------
        snapshots : list of int
            List of snapshot numbers to process
        hdf5_file : str
            Output HDF5 file path
        x3, x2, x1 : int, optional
            Index along the axis to slice (exactly one must be specified)
        fields : list, optional
            List of fields to extract
        """
        for i, snapshot in enumerate(snapshots):
            print(f"Processing snapshot {snapshot} ({i+1}/{len(snapshots)})")
            slice_data = self.slice(snapshot, x3=x3, x2=x2, x1=x1, fields=fields)
            mode = 'w' if i == 0 else 'a'
            slice_data.save_hdf5(hdf5_file, mode=mode)
            
        print(f"Saved {len(snapshots)} snapshots to {hdf5_file}")
    
    def _get_core_filename(self, snapshot, core_id):
        """"Construct the snapshot filename for a given core
        ID. Accounts for the different structures Athena uses
        that depend on the number of cores used in the run and
        the given core.

        If the working directory assigned to `Athena.datadir` is
        named `./output/`, this function returns the following
        relative paths from the working directory:

        When 1 core is used:
            `./output/<job>.<snapshot>.bin`
        When >1 core is used:
            Core 0:
                `./output/id0/<job>.<snapshot>.bin`
            Other cores:
                `./output/id<core_id>/<job>-id<core_id>.<snapshot>.bin`
        
        Parameters
        ----------
        snaphot : int
            Snapshot number.
        core_id : int
            Core ID number.
        
        Returns
        -------
        str
            Relative path to the snapshot file for the specified core
            from the working directory.        
        """
        job = self.problem_id
        if self.inputs["Ncores"] == 1:
            # filename structure: ./output/<job>.<snapshot>.bin
            return os.path.join(self.datadir, f"{job}.{snapshot:0>4}.bin")
        if core_id == 0:
            # filename structure: ./output/id0/<job>.<snapshot>.bin
            return os.path.join(self.datadir, f"id{core_id}",f"{job}.{snapshot:0>4}.bin")
        else:
            # filename structure: ./output/id<core_id>/<job>-id<core_id>.<snapshot>.bin
            return os.path.join(self.datadir, f"id{core_id}",f"{job}-id{core_id}.{snapshot:0>4}.bin")

    def _initialize_slice_arrays(self, full_array_shape, fields):
        """Initialize arrays to hold sliced data_arrays for each specified field.
        
        Parameters
        ----------
        full_array_shape : tuple
            Shape of the 2D slice to be created.
        fields : list
            List of field names for which to initialize arrays.

        Returns
        -------
        dict
            Dictionary with data_arrays field names as keys and initialized arrays as values.
        """
        return {field: np.zeros(full_array_shape) for field in fields}
    
    def _construct_core_list(self, coreslice):
        """Create the list of cores containing the index of the 
        chosen slice.

        Parameters
        ----------
        coreslice : slice
            The slice object containing the indices of the
            corresponding cores.

        Returns
        -------
        list
            List of core IDs that contain the specified slice.
        """
        # Construct 3D array of core numbers 
        cores_array = np.arange(0, self.inputs["Ncores"]).reshape(
            (self.inputs["NGrid_x3"],
             self.inputs["NGrid_x2"],
             self.inputs["NGrid_x1"])
        )
        return cores_array[coreslice]

    def _core_location(self, x3, x2, x1, x3_0, x2_0, x1_0, dx3, dx2, dx1, axis):
        """Return a core's location within the monolithic data_arrays array.

        Parameters
        ----------
        x3 : ndarray
            Local z coordinate array.
        x2 : ndarray
            Local y coordinate array.
        x1 : ndarray
            Local x coordinate array.
        x3_0 : float
            Initial coordinate in the z-array.
        x2_0 : float
            Initial coordinate in the y-array.
        x1_0 : float
            Initial coordinate in the x-array.
        dx3 : float
            Cell size along z direction.
        dx2 : float
            Cell size along y direction.
        dx1 : float
            Cell size along x direction
        axis : str
            One of {"x3", "x2', "x1"} denoting the axis of the slice.

        Returns
        -------
        slice
            Indices used to place data_arrays from a core into the monolithic array.
        """

        # recurrence relations for each axis used to index monolithical dataset
        ix3_0 = np.int32(np.round((x3[0] - x3_0) / dx3))
        ix2_0 = np.int32(np.round((x2[0] - x2_0) / dx2))
        ix1_0 = np.int32(np.round((x1[0] - x1_0) / dx1))

        ix3_1 = ix3_0 + len(x3)
        ix2_1 = ix2_0 + len(x2)
        ix1_1 = ix1_0 + len(x1)

        # Location in full snapshot
        if axis == "x3":
            return np.s_[ix2_0:ix2_1, ix1_0:ix1_1]
        if axis == "x2":
            return np.s_[ix3_0:ix3_1, ix1_0:ix1_1]
        if axis == "x1":
            return np.s_[ix3_0:ix3_1, ix2_0:ix2_1]        

    def _construct_2Dslice(self, snapshot, x3index=None, x2index=None, x1index=None, fields=None):
        """Construct 2D slice from distributed binary files."""

        if fields is None:
            fields = ["rho", "rux1", "rux2", "rux3", "eng"]

        indices = np.array([x3index, x2index, x1index])
        
        # Test that exactly one coordinate is specified
        if np.sum(indices != None) != 1:
            raise ValueError("Must specify exactly one coordinate to create a 2D slice")
        
        # Record axis and plane information for use during slice construction
        axis = "x3" if x3index is not None else "x2" if x2index is not None else "x1"
        # plane = self.AXES[axis]["plane"]
        
        coreslice, local_slice, full_array_shape = self._construct_slice_shape(indices)
        corelist = self._construct_core_list(coreslice)

        # Initialize arrays for all fields
        arrays = self._initialize_slice_arrays(full_array_shape, fields)

        # Initialize coordinate tracking
        x1_all = x2_all = x3_all = None
        x10 = x20 = x30 = None
        dx1 = dx2 = dx3 = None
        t = dt = None

        # loop through cores
        for j, core_id in enumerate(corelist.flatten()):
            filename = self._get_core_filename(snapshot, core_id)
            print(f"  Reading {filename}")

            # Read data from this core
            x3, x2, x1, *data_arrays, t, dt = read_2Dfrom3D(filename, local_slice)

            # Package data_arrays into a dictionary
            core_data = dict(zip(fields, data_arrays))
            
            # Initialize reference coordinates on first iteration            
            if j == 0:
                x30, x20, x10 = x3[0], x2[0], x1[0]
                dx3 = x3[1] - x3[0] if len(x3) > 1 else 0
                dx2 = x2[1] - x2[0] if len(x2) > 1 else 0
                dx1 = x1[1] - x1[0] if len(x1) > 1 else 0

                x3_all = x3.copy()
                x2_all = x2.copy()
                x1_all = x1.copy()
            else:
                # Extend coordinate arrays if needed
                if len(x3) > 0 and len(x3_all) > 0 and np.isclose(x3[0], x3_all[-1] + dx3):
                    x3_all = np.concatenate((x3_all, x3))
                if len(x2) > 0 and len(x2_all) > 0 and np.isclose(x2[0], x2_all[-1] + dx2):
                    x2_all = np.concatenate((x2_all, x2))
                if len(x1) > 0 and len(x1_all) > 0 and np.isclose(x1[0], x1_all[-1] + dx1): # np.isclose() used to handle machine precision deviations.
                    x1_all = np.concatenate((x1_all, x1))

            # Determine where this core's data goes in the full array
            core_loc = self._core_location(x3, x2, x1, x30, x20, x10, dx3, dx2, dx1, axis)

            # Insert core data_arrays into full arrays
            for field in fields:
                arrays[field][core_loc] = core_data[field]

        # Build coordinate dictionary for the sliced plane
        coords = {}
        if axis == "x3":
            coords["x2"] = x2_all
            coords["x1"] = x1_all
        if axis == "x2":
            coords["x3"] = x3_all
            coords["x1"] = x1_all
        if axis == "x1":
            coords["x3"] = x3_all
            coords["x2"] = x2_all

        return AthenaSlice(
        coords=coords,
        data=arrays,
        t=t,
        dt=dt,
        snapshot=snapshot,
        axis=axis,
        index=x3index or x2index or x1index
        )

    @classmethod
    def parse_athinput(self, file_athinput):
        """Retrieve values from athinput file.

        Parameters
        ----------
        file_athinput : str
            Path to the athinput file.

        Returns
        -------
        dict
            Dictionary containing the values from the athinput file.
        """

        # Read in the file
        with open(file_athinput, 'r') as f:
            lines = f.readlines()

        athinput_dict = {}
        skipped_keys = []

        for line in lines:
            if "=" in line and not line.startswith("#"):
                key = line.split("=")[0].strip()
                value_w_comment = line.split("=")[1].strip()
                value = value_w_comment.split("#")[0].strip()

                if key not in ATHINPUT_TYPES:
                    if key in SKIP_KEYS:
                        skipped_keys.append(key)
                        continue
                    else:
                        try:
                            athinput_dict[key] = int(value)
                            print(f"Attention: key `{key}` not in ATHINPUT_TYPES, adding as `{type(athinput_dict[key])}` type.")
                        except ValueError:
                            try:
                                athinput_dict[key] = float(value)
                                print(f"Attention: key `{key}` not in ATHINPUT_TYPES, adding as `{type(athinput_dict[key])}` type.")
                            except ValueError:
                                athinput_dict[key] = str(value)
                                print(f"Attention: key `{key}` not in ATHINPUT_TYPES, adding as `{type(athinput_dict[key])}` type.")

                else:
                    athinput_dict[key] = ATHINPUT_TYPES[key](value)

        print(f"Athinput file contains these keys but we didn't load them:\n   {skipped_keys}")

        # Add derived quantities
        athinput_dict["nx3_local"] = athinput_dict["Nx3"] // athinput_dict["NGrid_x3"]
        athinput_dict["nx2_local"] = athinput_dict["Nx2"] // athinput_dict["NGrid_x2"]
        athinput_dict["nx1_local"] = athinput_dict["Nx1"] // athinput_dict["NGrid_x1"]
        athinput_dict["Ncores"]    = athinput_dict["NGrid_x1"] * athinput_dict["NGrid_x2"] * athinput_dict["NGrid_x3"]

        return athinput_dict
           
    def _construct_slice_shape(self, coords):
        """Determine slice shape and core distribution."""
        # Get the slice of cores and define local coordinate
        # If x3 coordinate is spedified
        if coords[0] is not None:  # x3 coordinate specified
            x3coord = coords[0]
            nx3_local = self.inputs["nx3_local"]
            x3proc_slice = x3coord // nx3_local
            coreslice = np.s_[x3proc_slice, :, :]
            local_slice = np.s_[x3coord % nx3_local, :, :]
            full_array_shape = (self.inputs["Nx2"], self.inputs["Nx1"])

        if coords[1] is not None:  # x2 coordinate spedified
            x2coord = coords[1]
            nx2_local = self.inputs["nx2_local"]
            yproc_slice = x2coord // nx2_local
            coreslice = np.s_[:, yproc_slice, :]
            local_slice = np.s_[:, x2coord % nx2_local, :]
            full_array_shape = (self.inputs["Nx3"], self.inputs["Nx1"])

        if coords[2] is not None:  # x1 coordinate spedified
            x1coord = coords[2]
            nx1_local = self.inputs["nx1_local"]
            xproc_slice = x1coord // nx1_local
            coreslice = np.s_[:, :, xproc_slice]
            local_slice = np.s_[:, :, x1coord % nx1_local]
            full_array_shape = (self.inputs["Nx3"], self.inputs["Nx2"])

        return coreslice, local_slice, full_array_shape

def read_2Dfrom3D(binary_file, local_slice):
    """Read in data_arrays from a single Athena binary file.

    Parameters
    ----------
    file_binary : str
        Snapshot file in binary format.
    local_slice : slice
        Slice object defining the local slice of data_arrays to read from the core.

    Returns
    -------
    x3 : ndarray
        z-coordinates of the grid cells.
    x2 : ndarray
        y-coordinates of the grid cells.
    x1 : ndarray
        x-coordinates of the grid cells.
    rho : ndarray
        Density data_arrays array.
    rux1 : ndarray
        Momentum density in the x-direction data_arrays array.
    rux2 : ndarray
        Momentum density in the y-direction data_arrays array.
    rux3 : ndarray
        Momentum density in the z-direction data_arrays array.
    eng : ndarray
        Energy density data_arrays array.
    t : float
        Simulation time of the snapshot.
    dt : float
        Timestep of the snapshot.   
    """

    try:
        file = open(binary_file, 'rb')
    except:
        print('error opening', file.name)
        print('data_arrays must be <binary_dump>')
        raise SystemExit

    # Read header
    file.seek(0,2)
    eof = file.tell()
    file.seek(0,0)

    coordsys = np.fromfile(file, dtype=np.int32, count=1)[0]
    nx1, nx2, nx3, nvar, nscalars= np.fromfile(file,dtype=np.int32,count=5)
    selfgrav_boolean, particles_boolean = np.fromfile(file,dtype=np.int32,count=2)
    gamma1, cs = np.fromfile(file, dtype=np.float64, count=2)
    t,dt = np.fromfile(file, dtype=np.float64, count=2)

    # Data are stored in 1->3 order, but we'll return to 3->1 order to be consistent with
    #  the convention of x3 as the vertical direction.
    x1 = np.fromfile(file, dtype=np.float64, count=nx1)
    x2 = np.fromfile(file, dtype=np.float64, count=nx2)
    x3 = np.fromfile(file, dtype=np.float64, count=nx3)

    # Read and slice data
    localshape = (nx3, nx2, nx1)
    count = np.prod(localshape) #nx1*nx2*nx3

    # Same reasoning as above for the sequence of reading data arrays breaking the convention.
    rho  = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    rux1 = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    rux2 = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    rux3 = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    eng  = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]

    if file.tell() != eof:
        print('Error: Too few bytes read.')

    file.close()

    return x3, x2, x1, rho, rux3, rux2, rux1, eng, t, dt

def get_problem_id(datadir, Ncores):
    """Get the problem ID from the output data_arrays filename.
    If more than one core was used. it will account
    for the additional "id*" subdirectory structure.

    Parameters
    ----------
    datadir : str
        Parent directory containing Athena output data_arrays for the run.
        If Ncores > 1, this should be the directory containing the
        "id*" subdirectories.
    Ncores : int
        Number of cores used in the Athena run.
    """
    if Ncores == 1:
        jobname = os.listdir(datadir)[0].split('.')[0]
    else:
        jobname = os.listdir(os.path.join(datadir, 'id0'))[0].split('.')[0]

    return jobname

