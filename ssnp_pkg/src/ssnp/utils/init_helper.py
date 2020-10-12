import warnings


def test():
    from pycuda.compiler import preprocess_source
    from pycuda.driver import CompileError
    from pytools.prefork import ExecError

    def warn(s):
        warnings.warn(f"pycuda doesn't seem to work normally. {s}", stacklevel=5)

    try:
        preprocess_source('', [], 'nvcc')
    except CompileError as e:
        if e.stderr:
            warn(f"nvcc basic test output: {e.stderr}")
        else:
            warn(f"maybe no available C++ compiler found by nvcc?")
    except ExecError:
        warn(f"maybe nvcc is not in PATH?")
    except:
        warn(f"unknown error")


def autoinit():
    import pycuda.driver

    if pycuda.driver.Context.get_current() is None:
        # warnings.warn("no current context")
        import pycuda.autoinit


autoinit()
test()
