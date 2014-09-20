Fork of PBRT v2 for COMP 408 at VUW.
Not really a fork because of build system changes and removing all but the renderer source.

Original here: http://github.com/mmp/pbrt-v2.

PBRT for COMP 408
=================

This is just the PBRT renderer (no tools / exporters), with a CMake build system.

The `cmake` branch has the CMake and pbrt-v2 source (modified just enough to build nicely).
Other branches have my assignment in them.

Building on Linux
-----------------

From repository root:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
./build/bin/pbrt ./scenes/dragontest.pbrt
```

Building on Windows (VS2013, x64)
---------------------------------

From repository root, in cmd:
```
mkdir build
cd build
cmake .. -G "Visual Studio 12 Win64"
```

In VS:
- Set ```pbrt``` as the startup project
- Set its working directory to ```$(SolutionDir)..``` (all configs)
- Set its command arguments to ```./scenes/dragontest.pbrt``` (all configs)
- Switch to release config
- Run dat shiznit

