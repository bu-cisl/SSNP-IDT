import warnings
import platform


def pycuda_test():
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
    except BaseException as e:
        warn(f"unknown error: {e!r}")


def autoinit():
    import pycuda.driver

    if pycuda.driver.Context.get_current() is None:
        # warnings.warn("no current context")
        import pycuda.autoinit


def patch_compiler_cache():
    import pycuda.compiler
    code = pycuda.compiler.compile_plain.__code__
    code_consts = list(code.co_consts)
    if code_consts.count('#include') != 1:
        warnings.warn(
            "cannot apply the patch to eliminate preprocessing before caching, since pycuda.compiler.compile_plain has been changed")
        return
    code_consts[code_consts.index('#include')] = '!!NOTEXISTED#@!!'
    pycuda.compiler.compile_plain.__code__ = code.replace(co_consts=tuple(code_consts))


autoinit()
pycuda_test()
if platform.system() == 'Windows':
    patch_compiler_cache()
