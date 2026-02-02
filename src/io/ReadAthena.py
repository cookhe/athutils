"""Functions to read Athena files."""

import os
import numpy as np
from dataclasses import dataclass

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

    def save(self, filename):
        np.savez_compressed(
            filename,
            x=self.x,
            y=self.y,
            z=self.z
            t=self.t,
            dt=self.dt,
            snapshot=self.snapshot,
            axis=self.axis,
            index=self.index,
            **self.data
        )

class Athena:
    def __init__(self,
                 file_athinput,
                 datadir
                ):
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
        return self._construct_2Dslice(
            datadir=self.datadir,
            snapshot=snapshot,
            x3index=x3,
            x2index=x2,
            x1index=x1,
        )
    
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
        """Initialize arrays to hold sliced data for each specified field.
        
        Parameters
        ----------
        full_array_shape : tuple
            Shape of the 2D slice to be created.
        fields : list
            List of field names for which to initialize arrays.

        Returns
        -------
        dict
            Dictionary with data field names as keys and initialized arrays as values.
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
        cores_array = np.arange(0,self.inputs["Ncores"]).reshape(
            (self.inputs["Nzcores"],
             self.inputs["Nycores"],
             self.inputs["Nxcores"])
            )
        # Create 2D array of the cores containing the desired slice
        return cores_array[coreslice]

    def _core_location(self, z, y, x, z0, y0, x0, dz, dy, dx, axis):
        """Return a core's location within the monolithic data array.

        Parameters
        ----------
        z : ndarray
            Local z coordinate array.
        y : ndarray
            Local y coordinate array.
        x : ndarray
            Local x coordinate array.
        z0 : float
            Initial coordinate in the z-array.
        y0 : float
            Initial coordinate in the y-array.
        x0 : float
            Initial coordinate in the x-array.
        dz : float
            Cell size along z direction.
        dy : float
            Cell size along y direction.
        dx : float
            Cell size along x direction
        axis : str
            One of {"x3", "x2', "x1"} denoting the axis of the slice.

        Returns
        -------
        slice
            Indices used to place data from a core into the monolithic array.
        """

        # recurrence relations for each axis used to index monolithical dataset
        iz0 = np.int32(np.round((z[0]-z0)/dz)) # removed the 12 order, becsuse rounded improperly
        iy0 = np.int32(np.round((y[0]-y0)/dy, 12))
        ix0 = np.int32(np.round((x[0]-x0)/dx, 12))

        iz1 = iz0 + len(z)
        iy1 = iy0 + len(y)
        ix1 = ix0 + len(x)

        # Location in full snapshot
        if axis == "x3":
            return np.s_[iy0:iy1,ix0:ix1]
        if axis == "x2":
            return np.s_[iz0:iz1,ix0:ix1]
        if axis == "x1":
            return np.s_[iz0:iz1,iy0:iy1]
        
    def _insert_core_slice(self, arrays, core_data, coreloc):
        """Insert data from a core into the monolithic data arrays.

        Parameters
        ----------
        arrays : dict
            Dictionary containing the monolithic data arrays.
        core_data : dict
            Dictionary containing the data arrays from the core.
        coreloc : slice
            Slice object defining the location in the monolithic arrays
            where the core data should be inserted.
        """
        for field, arr in arrays.items():
            arr[coreloc] = core_data[field]
        

    def _construct_2Dslice(self, snapshot, x3index=None, x2index=None, x1index=None):
    
        indices = np.array([x3index, x2index, x1index])
        
        # Test that exactly one coordinate is specified
        if np.sum(indices != None) != 1:
            raise ValueError("Must specify exactly one coordinate to create a 2D slice")
        
        # Record axis and plane information for use during slice construction
        axis = "x3" if x3index is not None else "x2" if x2index is not None else "x1"
        plane = self.AXES[axis]["plane"]
        
        coreslice, local_slice, full_array_shape = self._construct_slice_shape(indices)

        # Get list of cores containing the slice
        corelist = self._construct_core_list(coreslice)

        fields = ["rho", "rux", "ruy", "ruz", "eng"]
        arrays = self._initialize_slice_arrays(full_array_shape, fields)

        # loop through cores
        for j, core_id in enumerate(corelist.flatten()):
            filename = self._get_core_filename(snapshot, core_id)
            print(filename)

            # z, y, x, rho, rux, ruy, ruz, eng, t, dt = read_binary(filename, local_slice)
            z, y, x, *data, t, dt = read_2Dfrom3D(filename, local_slice)
            core_data = dict(zip(fields, data))
            
            
            core_local_slice = self._core_location(z, y, x,
                                                   z[0], y[0], x[0],
                                                   z[1]-z[0], y[1]-y[0], x[1]-x[0],
                                                   axis= "x3" if x3index is not None else "x2" if x2index is not None else "x1")
        
            
            if j == 0:
                # x0=x[0]
                # y0=y[0]
                # z0=z[0]
                # dx=x[1]-x[0]
                # dy=y[1]-y[0]
                # dz=z[1]-z[0]

                # initiate the full-length x,y,z arrays
                # x_all = x
                # y_all = y
                # z_all = z

                # add next set of coordinates if first value follows existing (limiting sum to machine precision)
            if np.isclose(x[0], x_all[-1] + dx): # np.isclose() used to handle machine precision deviations.
                x_all = np.concatenate((x_all,x), axis=0)

            if np.isclose(y[0], y_all[-1] + dy):
                y_all = np.concatenate((y_all,y), axis=0)

            if np.isclose(z[0], z_all[-1] + dz):
                z_all = np.concatenate((z_all,z), axis=0)

            # Populate coordinates for the sliced plane
            coords = {}
            if axis == "x3":
                coords["x2"] = x2_all
                coords["x1"] = x1_all
            if axis == "x2":
                coords["x3"] = z3_all
                coords["x1"] = x1_all
            if axis == "x1":
                coords["x3"] = z3_all
                coords["x2"] = x2_all

            # # now populate the *_all arrays to recreate the monolithical dataset
            rho_all[coreloc] = rho
            rux_all[coreloc] = rux
            ruy_all[coreloc] = ruy
            ruz_all[coreloc] = ruz
            eng_all[coreloc] = eng
            

        return AthenaSlice(
        coords=coords,
        data=arrays,
        t=t,
        dt=dt,
        snapshot=snapshot_num,
        axis="x3" if x3index is not None else "x2" if x2index is not None else "x1",
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

        for line in lines:
            if "=" in line and not line.startswith("#"):
                key = line.split("=")[0].strip()
                value_w_comment = line.split("=")[1].strip()
                value = value_w_comment.split("#")[0].strip()

                if key not in ATHINPUT_TYPES:
                # print(f"WARNING: key `{key}` not in ATHINPUT_TYPES, adding as float type for now, \
                #       which may introduce unexpected behavior if {key} is a different type.")
                    if key in SKIP_KEYS:
                        # print(f"Skipping key `{key}`.")
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
                print(f"Skipped keys: {SKIP_KEYS}")

        # Add derived quantities
        athinput_dict["nx3_local"] = athinput_dict["Nx3"] // athinput_dict["NGrid_x3"]
        athinput_dict["nx2_local"] = athinput_dict["Nx2"] // athinput_dict["NGrid_x2"]
        athinput_dict["nx1_local"] = athinput_dict["Nx1"] // athinput_dict["NGrid_x1"]
        athinput_dict["Ncores"]    = athinput_dict["NGrid_x1"] * athinput_dict["NGrid_x2"] * athinput_dict["NGrid_x3"]

        return athinput_dict
    

        
    def _construct_slice_shape(self, coords):
        # Get the slice of cores and define local coordinate
        # If x3 coordinate is spedified
        if coords[0] != None:
            x3coord = coords[0]
            nx3_local = self.inputs["nx3_local"]
            zproc_slice = x3coord // nx3_local
            coreslice = np.s_[zproc_slice,:,:]
            local_slice = np.s_[x3coord % nx3_local,:,:]
            full_array_shape = (
                self.inputs["Nx2"],
                self.inputs["Nx1"]
            )
        # If x2 coordinate is spedified
        if coords[1] != None:
            x2coord = coords[1]
            nx2_local = self.inputs["nx2_local"]
            yproc_slice = x2coord // nx2_local
            coreslice = np.s_[:,yproc_slice,:]
            local_slice = np.s_[:,x2coord%nx2_local,:]
            full_array_shape = (
                self.inputs["Nx3"],
                self.inputs["Nx1"]
            )
        # If x1 coordinate is spedified
        if coords[2] != None:
            x1coord = coords[2]
            nx1_local = self.inputs["nx1_local"]
            xproc_slice = x1coord // nx1_local
            coreslice = np.s_[:,:,xproc_slice]
            local_slice = np.s_[:,:,x1coord%nx1_local]
            full_array_shape = (
                self.inputs["Nx3"],
                self.inputs["Nx2"]
            )

        return coreslice, local_slice, full_array_shape

def read_2Dfrom3D(binary_file, local_slice):
    """Read in data from a single Athena binary file.

    Parameters
    ----------
    file_binary : str
        Snapshot file in binary format.
    local_slice : slice
        Slice object defining the local slice of data to read from the core.

    Returns
    -------
    z : ndarray
        z-coordinates of the grid cells.
    y : ndarray
        y-coordinates of the grid cells.
    x : ndarray
        x-coordinates of the grid cells.
    rho : ndarray
        Density data array.
    rux : ndarray
        Momentum density in the x-direction data array.
    ruy : ndarray
        Momentum density in the y-direction data array.
    ruz : ndarray
        Momentum density in the z-direction data array.
    eng : ndarray
        Energy density data array.
    t : float
        Simulation time of the snapshot.
    dt : float
        Timestep of the snapshot.   
    """

    # Confirm it's a C binary
    try:
        file = open(binary_file, 'rb')
    except:
        print('error opening', file.name)
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

    x = np.fromfile(file, dtype=np.float64, count=nx)
    y = np.fromfile(file, dtype=np.float64, count=ny)
    z = np.fromfile(file, dtype=np.float64, count=nz)

    localshape = (nz, ny, nx)
    count = np.prod(localshape) #nx*ny*nz

    rho = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    rux = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    ruy = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    ruz = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]
    eng = np.fromfile(file, dtype=np.float64, count=count).reshape(localshape)[local_slice]

    if file.tell() != eof:
        print('Error: Too few bytes read.')

    file.close()

    return z, y, x, rho, rux, ruy, ruz, eng, t, dt

def get_problem_id(datadir, Ncores):
    """Get the problem ID from the output data filename.
    If more than one core was used. it will account
    for the additional "id*" subdirectory structure.

    Parameters
    ----------
    datadir : str
        Parent directory containing Athena output data for the run.
        If Ncores > 1, this should be the directory containing the
        "id*" subdirectories.
    Ncores : int
        Number of cores used in the Athena run.
    """
    if Ncores == 1:
        jobname = os.listdir(datadir)[0].split('.')[0]
    else:
        jobname = os.listdir(os.path.join(datadir,'id0'))[0].split('.')[0]

    return jobname

