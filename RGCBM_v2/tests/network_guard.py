"""Linux-only validation wrapper: deny Internet sockets for a child Python process.
Requires the system libseccomp library. Not part of Windows environment setup.
Usage: python tests/network_guard.py /path/to/python tests/validate_offline_notebook.py --output ...
"""
import ctypes
import ctypes.util
import errno
import os
import socket
import sys

lib = ctypes.CDLL(ctypes.util.find_library('seccomp'), use_errno=True)
class Arg(ctypes.Structure):
    _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_uint),('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]
lib.seccomp_init.argtypes=[ctypes.c_uint32]; lib.seccomp_init.restype=ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Arg)]
lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=lib.seccomp_init(0x7fff0000)
if not ctx:raise RuntimeError('Cannot initialize offline syscall guard')
for name in [b'socket',b'socketpair']:
    number=lib.seccomp_syscall_resolve_name(name)
    for family in [socket.AF_INET,socket.AF_INET6,socket.AF_PACKET]:
        condition=Arg(0,4,family,0)
        if lib.seccomp_rule_add_array(ctx,0x50000|errno.EPERM,number,1,ctypes.byref(condition))!=0:raise RuntimeError('Cannot add offline rule')
if lib.seccomp_load(ctx)!=0:raise RuntimeError('Cannot enforce offline syscall guard')
lib.seccomp_release(ctx)
for family in [socket.AF_INET,socket.AF_INET6]:
    try: socket.socket(family)
    except PermissionError:pass
    else:raise RuntimeError('Offline guard did not block Internet sockets')
print('OFFLINE GUARD: IPv4/IPv6/packet sockets blocked by inherited kernel seccomp rules; Unix IPC allowed.',flush=True)
os.environ.update(PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CONFIG_FILE='/dev/null',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2',MPLBACKEND='Agg',PYTHONNOUSERSITE='1')
os.execvp(sys.argv[1],sys.argv[1:])
