import platform

if platform.system() == 'Windows':
    import os
    import logging
    import glob
    import pycuda.compiler

    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']
    # remove code below if you have valid C compiler in `PATH` already
    CL_PATH = glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio"
                        r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
    if CL_PATH:
        CL_PATH = max(CL_PATH)
        logging.info(f"MSVC compiler found: {CL_PATH}")
        os.environ['PATH'] += ";" + CL_PATH[:-7]
    else:
        logging.warning("Cannot find MSVC compiler")
