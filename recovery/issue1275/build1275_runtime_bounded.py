import multiprocessing, concurrent.futures, runpy, sys
# Bound nested upstream runtime builds without changing its source files.
multiprocessing.cpu_count = lambda: 8
_original = concurrent.futures.ThreadPoolExecutor
class SerialRuntimeBuilds(_original):
    def __init__(self, *args, **kwargs):
        kwargs['max_workers'] = 1
        super().__init__(**kwargs)
concurrent.futures.ThreadPoolExecutor = SerialRuntimeBuilds
sys.argv = ['build_runtimes', '--platforms', 'a2a3', 'a5']
runpy.run_module('simpler_setup.build_runtimes', run_name='__main__')
