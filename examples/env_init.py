import platform

if platform.system() == 'Windows':
    import os
    import logging
    import glob
    import shutil
    import pycuda.compiler

    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']
    # try to find a MSVC compiler if 'cl.exe' is not in PATH
    if shutil.which('cl') is None:
        old_path = os.environ['PATH']
        CL_PATH = glob.glob(r"C:\Program Files*\Microsoft Visual Studio"
                            r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
        if CL_PATH:
            CL_PATH = max(CL_PATH)
            os.environ['PATH'] += ";" + CL_PATH[:-7]

        if CL_PATH and shutil.which('cl') is not None:
            logging.info(f"MSVC compiler found and added to PATH: {CL_PATH}")
        else:
            logging.warning("Cannot find MSVC compiler")
            os.environ['PATH'] = old_path
