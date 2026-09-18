"""Data-only IPC and bounded execution in restricted Linux Docker containers.

No host execution fallback. No host files or secrets are mounted/passed. Containers
share a kernel: this is not a VM boundary. See docs/ISOLATION_BOUNDARY.md.
"""
from __future__ import annotations
import ast
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    SUCCESS = 'SUCCESS'
    EXCEPTION = 'EXCEPTION'
    TIMEOUT = 'TIMEOUT'
    CRASH = 'CRASH'
    OVERSIZED_OUTPUT = 'OVERSIZED_OUTPUT'
    CANCELLED = 'CANCELLED'
    INFRASTRUCTURE_FAILURE = 'INFRASTRUCTURE_FAILURE'


@dataclass
class IsolatedExecutionResult:
    status: ExecutionStatus
    value: Any = None
    exception_type: str | None = None
    exception_message: str | None = None
    wall_time_sec: float = 0.
    output_bytes: int = 0
    diagnostic: str = ''

    def is_success(self):
        return self.status == ExecutionStatus.SUCCESS

    def has_matching_exception(self, expected_exc_type_name):
        return self.status == ExecutionStatus.EXCEPTION and self.exception_type == expected_exc_type_name


def validate_data(value, depth=0):
    """Only bounded JSON primitives; no object hooks or executable codecs."""
    if depth > 32: raise ValueError('JSON nesting exceeds 32')
    kind = type(value)
    if value is None or kind in (bool, str): return
    if kind is int and value.bit_length() <= 256: return
    if kind is float and math.isfinite(value): return
    if kind is list:
        for item in value: validate_data(item, depth+1)
        return
    if kind is dict and all(type(k) is str for k in value):
        for item in value.values(): validate_data(item, depth+1)
        return
    raise ValueError('Unsupported data type or numeric range')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def decode_response(raw, limit):
    if len(raw) > limit: raise ValueError('Response exceeds limit')
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs)
    validate_data(data)
    if type(data) is not dict: raise ValueError('Response must be an object')
    status = data.get('status')
    if status == 'SUCCESS' and set(data) == {'status', 'value'}:
        return IsolatedExecutionResult(ExecutionStatus.SUCCESS, value=data['value'], diagnostic='Pass')
    if status == 'EXCEPTION' and set(data) == {'status','exception_type','exception_message'}:
        if not all(type(data[k]) is str for k in ('exception_type','exception_message')):
            raise ValueError('Invalid exception schema')
        return IsolatedExecutionResult(ExecutionStatus.EXCEPTION,
            exception_type=data['exception_type'], exception_message=data['exception_message'],
            diagnostic='Candidate exception')
    if status == 'UNSUPPORTED_RESULT' and set(data) == {'status'}:
        return IsolatedExecutionResult(ExecutionStatus.INFRASTRUCTURE_FAILURE,
                                       diagnostic='Unsupported candidate result type')
    raise ValueError('Invalid worker response schema')


# Sent as a Python argument, not as a host mount. All worker output is untrusted.
WORKER = r'''
import json, sys, math
def valid(v, d=0):
    if d>32: raise ValueError('depth')
    t=type(v)
    if v is None or t in (bool,str): return
    if t is int and v.bit_length()<=256: return
    if t is float and math.isfinite(v): return
    if t is list:
        for x in v: valid(x,d+1)
        return
    if t is dict and all(type(k) is str for k in v):
        for x in v.values(): valid(x,d+1)
        return
    raise ValueError('unsupported data')
p=json.loads(sys.stdin.buffer.read(262145))
ns={'__name__':'candidate'}
try:
    exec(p['code'],ns)
    result=ns[p['func_name']](**p['kwargs'])
except Exception as e:
    response={'status':'EXCEPTION','exception_type':type(e).__module__+'.'+type(e).__qualname__,'exception_message':str(e)}
else:
    try:
        valid(result)
        response={'status':'SUCCESS','value':result}
    except Exception:
        response={'status':'UNSUPPORTED_RESULT'}
sys.stdout.write(json.dumps(response,allow_nan=False))
sys.stdout.flush()
'''


class ProcessIsolatedRunner:
    """Requires Linux Docker, including Linux Docker Desktop on Windows."""
    IMAGE = 'python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'
    enforces_isolation = True

    def __init__(self, timeout_sec=2., max_output_bytes=65536, *, memory_mb=64,
                 pids_limit=16, docker_executable=None):
        if not math.isfinite(timeout_sec) or timeout_sec <= 0 or max_output_bytes < 1:
            raise ValueError('Timeout and output limit must be positive')
        if memory_mb < 32 or pids_limit < 1: raise ValueError('Invalid resource limit')
        self.timeout_sec, self.max_output_bytes = float(timeout_sec), int(max_output_bytes)
        self.memory_mb, self.pids_limit = int(memory_mb), int(pids_limit)
        self.docker = docker_executable or (shutil.which('docker.exe') if os.name == 'nt' else None) or shutil.which('docker')

    def build_sanitized_env(self):
        # Trusted Docker CLI only. None of these host values enters the container.
        allowed = {'SYSTEMROOT','WINDIR','TEMP','TMP','PATH'}
        return {k:v for k,v in os.environ.items() if k.upper() in allowed}

    def _control(self, args):
        if not self.docker: raise RuntimeError('Docker unavailable; no unsafe fallback')
        return subprocess.run(self._command(args), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            env=self.build_sanitized_env())

    def _command(self, args):
        # Do not honor a saved remote context or DOCKER_HOST for private candidates.
        endpoint = 'npipe:////./pipe/docker_engine' if os.name == 'nt' else 'unix:///var/run/docker.sock'
        return [self.docker, '--host='+endpoint, *args]

    def _preflight(self):
        r = self._control(['info','--format','{{json .}}'])
        if r.returncode: raise RuntimeError('Linux Docker unavailable; no unsafe fallback')
        info = json.loads(r.stdout)
        if info.get('OSType') != 'linux' or not all(info.get(k) for k in ('MemoryLimit','SwapLimit','PidsLimit')):
            raise RuntimeError('Linux memory/swap/PID enforcement unavailable')
        r = self._control(['image','inspect','--format','{{.Id}}',self.IMAGE])
        if r.returncode: raise RuntimeError('Pinned image unavailable; explicitly pull before execution')

    def _create_args(self, name):
        return ['create','--name',name,'--label','governed-agent.isolation=step2',
            '--pull=never','--network=none','--read-only','--cap-drop=ALL',
            '--security-opt=no-new-privileges','--user=65534:65534',
            '--memory='+str(self.memory_mb)+'m','--memory-swap='+str(self.memory_mb)+'m',
            '--pids-limit='+str(self.pids_limit),'--cpus=1','--ulimit=nofile=64:64',
            '--ulimit=core=0:0','--log-driver=none','--workdir=/tmp',
            '--env=LANG=C.UTF-8','--env=PYTHONHASHSEED=0','-i',self.IMAGE,
            'python','-I','-S','-B','-c',WORKER]

    def _exchange(self, name, payload, cancel_event):
        # Only the trusted Docker CLI runs on the host. We never call killpg.
        proc = subprocess.Popen(self._command(['start','--attach','--interactive',name]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.build_sanitized_env(), start_new_session=(os.name != 'nt'))
        buffers, count = [bytearray(),bytearray()], [0]
        lock, overflow = threading.Lock(), threading.Event()
        def read(pipe,index):
            try:
                while True:
                    chunk = pipe.read1(4096)
                    if not chunk: break
                    with lock:
                        available = max(0,self.max_output_bytes-count[0])
                        buffers[index].extend(chunk[:available])
                        count[0] += len(chunk)
                        if count[0] > self.max_output_bytes: overflow.set()
            except (OSError,ValueError): pass
        def write():
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except (BrokenPipeError,OSError,ValueError): pass
        readers = [threading.Thread(target=read,args=(pipe,i),daemon=True)
                   for i,pipe in enumerate((proc.stdout,proc.stderr))]
        writer = threading.Thread(target=write,daemon=True)
        for t in readers+[writer]: t.start()
        deadline, forced = time.monotonic()+self.timeout_sec, None
        try:
            while True:
                if overflow.is_set(): forced = ExecutionStatus.OVERSIZED_OUTPUT; break
                if cancel_event is not None and cancel_event.is_set(): forced = ExecutionStatus.CANCELLED; break
                if time.monotonic() >= deadline: forced = ExecutionStatus.TIMEOUT; break
                if proc.poll() is not None and not any(t.is_alive() for t in readers): break
                time.sleep(.005)
        finally:
            # Container removal destroys descendants on success and every failure.
            try:
                cleanup = self._control(['rm','--force',name])
            finally:
                if proc.poll() is None: proc.kill()
                proc.wait(timeout=5)
                for t in readers+[writer]: t.join(timeout=2)
                for pipe in (proc.stdin,proc.stdout,proc.stderr): pipe.close()
            if cleanup.returncode: raise RuntimeError('Cleanup failed; admission forbidden')
        if forced is not None:
            return IsolatedExecutionResult(forced,output_bytes=count[0],diagnostic=forced.value)
        if overflow.is_set():
            return IsolatedExecutionResult(ExecutionStatus.OVERSIZED_OUTPUT,output_bytes=count[0])
        if proc.returncode:
            return IsolatedExecutionResult(ExecutionStatus.CRASH,output_bytes=count[0],
                diagnostic=f'Worker crashed with exit code {proc.returncode}')
        result = decode_response(bytes(buffers[0]),self.max_output_bytes)
        result.output_bytes = count[0]
        return result

    def run_candidate(self, code, func_name=None, kwargs=None, *, cancel_event=None):
        start, name = time.monotonic(), 'governed-step2-'+uuid.uuid4().hex
        created = False
        try:
            if cancel_event is not None and cancel_event.is_set():
                return IsolatedExecutionResult(ExecutionStatus.CANCELLED)
            if type(code) is not str or len(code.encode('utf-8')) > 65536:
                raise ValueError('Invalid or oversized source')
            if not func_name or func_name == '<lambda>':
                funcs = [n.name for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
                if not funcs: raise ValueError('No module-level function')
                func_name = funcs[0]
            kwargs = {} if kwargs is None else kwargs
            if type(kwargs) is not dict: raise ValueError('Arguments must be an object')
            validate_data(kwargs)
            payload = json.dumps(dict(code=code,func_name=func_name,kwargs=kwargs),allow_nan=False).encode()
            if len(payload)>262144: raise ValueError('Input exceeds 256 KiB')
            self._preflight()
            # Set before create: cleanup also covers a timed-out create request.
            created = True
            r = self._control(self._create_args(name))
            if r.returncode: raise RuntimeError('Cannot create restricted container')
            result = self._exchange(name,payload,cancel_event)
            created = False
        except Exception as exc:
            result = IsolatedExecutionResult(ExecutionStatus.INFRASTRUCTURE_FAILURE,diagnostic=str(exc)[:300])
        finally:
            if created:
                try: self._control(['rm','--force',name])
                except Exception: pass
        result.wall_time_sec = time.monotonic()-start
        return result
