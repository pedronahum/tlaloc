"""§0.4.475 (H6a) — the PJRT C API, bound from Python with `ctypes` alone.

WHY THIS FILE EXISTS
====================

`tlaloc_serve.py` (H3a) runs a Tlaloc serving artifact in a process with no
JVM in it — but it reaches PJRT through **jaxlib**: `jax._src.xla_bridge`
for the backend, `jaxlib.mlir` for the module, `jaxlib._jax.CompileOptions`
for the compile. That is a 500 MB dependency, a CUDA-12 wheel family and a
version-locked MLIR python extension, dragged in to do a thing the PJRT C
ABI already exposes: load a plugin, compile a StableHLO string, put bytes on
a device, execute, take bytes back.

This module is that same surface with **nothing but the standard library**.
No jax. No jaxlib. No torch. No numpy. The only external artifact it needs
is a PJRT plugin `.so`, and a plugin `.so` can live **anywhere** — inside
somebody else's jax install, in `/lib/libtpu.so` on a Cloud TPU VM, or in a
standalone directory a deployment ships. That is exactly what makes the
serving runtime dependency-free: the serving process needs a driver and a
plugin file, not a framework.

WHAT IT IS A TRANSLATION OF
===========================

`runtime-pjrt/src/jvmMain/kotlin/io/tlaloc/runtime/pjrt/ffm/PjrtFfm.kt` —
Tlaloc's Kotlin FFM binding, which has been carrying every PJRT lane in this
repository since §0.4.303 and is certified against real XLA on the GB10. This
file is a **mechanical mirror** of it, function for function and offset for
offset, and every non-obvious constant here carries the § number of the
commit that learned it the hard way. Nothing was re-derived from the header;
re-deriving it would have meant re-learning §0.4.333 (the reboot) and
§0.4.304 (the Check-fail) by repeating them.

ONE STRUCTURAL DIFFERENCE, AND IT IS IN OUR FAVOUR
==================================================

The Kotlin has to spell C's padding out by hand — `MemoryLayout` lays fields
down exactly where you put them, so `PJRT_ExecuteOptions` carries an explicit
`paddingLayout(4)` after the i32 `launch_id` to get the next pointer back to
8-byte alignment. `ctypes` follows the C compiler's own alignment rules and
inserts that padding itself. So the struct definitions below mirror the
*header*, not the Kotlin's hand-padding — and to make sure the two agree,
every struct's `ctypes.sizeof` is asserted against the size the Kotlin
computes, at import time (`_assert_layouts`), and reported to the JVM by
`run_pjrt_ctypes_check.py --layouts` so the comparison is a test rather
than a comment.

Single module, not a package, deliberately: the Kotlin original is one file
whose value is that you can read the whole PJRT contract top to bottom, and
the only reason to split this would be to make `git diff` against that file
harder.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# PJRT_Api function-pointer offsets — §0.4.303, mirrored verbatim.
#
# `PJRT_Api` opens with (struct_size: size_t, extension_start: ptr,
# pjrt_api_version: PJRT_Api_Version) where PJRT_Api_Version is
# (size_t + ptr + 2 ints) = 24 bytes. First function pointer therefore at
# 8 + 8 + 24 = 40, each subsequent one +8 (LP64: aarch64 and x86_64 alike).
# ---------------------------------------------------------------------------

OFFSET_PJRT_Error_Destroy = 40
OFFSET_PJRT_Error_Message = 48
OFFSET_PJRT_Event_Destroy = 80
OFFSET_PJRT_Event_Await = 104
OFFSET_PJRT_Client_Create = 120
OFFSET_PJRT_Client_Destroy = 128
OFFSET_PJRT_Client_PlatformName = 136
OFFSET_PJRT_Client_AddressableDevices = 168
OFFSET_PJRT_Client_Compile = 200
OFFSET_PJRT_Client_BufferFromHostBuffer = 216
OFFSET_PJRT_Executable_Destroy = 360
OFFSET_PJRT_Executable_NumOutputs = 392
OFFSET_PJRT_LoadedExecutable_Destroy = 440
OFFSET_PJRT_LoadedExecutable_GetExecutable = 448
OFFSET_PJRT_LoadedExecutable_Execute = 480
OFFSET_PJRT_Buffer_Destroy = 504
OFFSET_PJRT_Buffer_OnDeviceSizeInBytes = 552
OFFSET_PJRT_Buffer_ToHostBuffer = 600

# The PJRT C API major version these bindings are written against, and the
# number of `PJRT_Api` bytes a plugin must provide: through the last function
# pointer read above. Mirrors `PjrtFfm.checkApiCompatible`.
PJRT_API_MAJOR_SUPPORTED = 0
PJRT_API_HIGHEST_OFFSET_USED = OFFSET_PJRT_Buffer_ToHostBuffer
PJRT_API_MIN_STRUCT_SIZE = PJRT_API_HIGHEST_OFFSET_USED + 8

# `PJRT_Api` header: struct_size @0, extension_start @8, then the embedded
# PJRT_Api_Version (struct_size @16, extension_start @24, major @32, minor @36).
_OFF_API_STRUCT_SIZE = 0
_OFF_API_VERSION_MAJOR = 32
_OFF_API_VERSION_MINOR = 36

# PJRT_Buffer_Type (xla/pjrt/c/pjrt_c_api.h:907). F32 = 11 and F64 = 12 are
# §0.4.303/§0.4.354; BF16 = 13 sits directly after F64 and is §0.4.457 (G1c).
#
# S32 = 4 is the ONE code in this file the Kotlin does not carry — the JVM
# side has never staged an integer buffer, so there was nothing to mirror.
# It is read off the same enum the Kotlin cites (INVALID=0, PRED=1, S8..S64
# = 2..5, U8..U64 = 6..9, F16=10, F32=11, F64=12, BF16=13), and it is
# certified here rather than assumed: the i32 lane of the certification
# round-trips a buffer through a real XLA executable and compares exact
# integers, which a wrong type code cannot survive.
PJRT_BUFFER_TYPE_S32 = 4
PJRT_BUFFER_TYPE_F32 = 11
PJRT_BUFFER_TYPE_F64 = 12
PJRT_BUFFER_TYPE_BF16 = 13

# §0.4.480 — the raw-bytes staging table: Tlaloc dtype name -> (PJRT type
# code, bytes per element). Keys are `io.tlaloc.core.DType.name` verbatim, the
# same spelling `tlaloc_serve._STAGE` uses, so an artifact cannot name a dtype
# one of the two tables understands and the other does not.
RAW_DTYPES = {
    "f32": (PJRT_BUFFER_TYPE_F32, 4),
    "i32": (PJRT_BUFFER_TYPE_S32, 4),
    "bf16": (PJRT_BUFFER_TYPE_BF16, 2),
}

# PJRT_NamedValue_Type: kString=0, kInt64=1, kInt64List=2, kFloat=3, kBool=4.
PJRT_NAMED_VALUE_TYPE_INT64 = 1
PJRT_NAMED_VALUE_TYPE_FLOAT = 3
PJRT_NAMED_VALUE_TYPE_BOOL = 4

# PJRT_HostBufferSemantics: 0 = kImmutableOnlyDuringCall. The simplest
# semantics — PJRT may not hold the host buffer past the call, so the host
# array can be freed as soon as it returns (§0.4.304).
HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL = 0

# §0.4.304's hand-encoded minimal CompileOptionsProto, re-verified for TPU in
# §0.4.459. XLA's PJRT compile path defaults num_replicas / num_partitions to
# 0 in C++ (not 1), and ParseDeviceAssignmentCompileOptions **Check-fails** on
# replica_count > 0 unless they are set. These six bytes set both to 1:
#
#   0x1A = (3 << 3) | 2   CompileOptionsProto.executable_build_options,
#                         field 3, wire type 2 (length-delimited)
#   0x04                  embedded message length
#     0x20 = (4 << 3) | 0 ExecutableBuildOptionsProto.num_replicas, varint
#     0x01                value 1
#     0x28 = (5 << 3) | 0 ExecutableBuildOptionsProto.num_partitions, varint
#     0x01                value 1
#
# Protobuf wire format is backend-agnostic and every PJRT plugin deserialises
# the same message, so these bytes are as correct on a TPU as on this GB10.
COMPILE_OPTIONS_PROTO_BYTES = bytes([0x1A, 0x04, 0x20, 0x01, 0x28, 0x01])


class PjrtError(RuntimeError):
    """A `PJRT_Error*` the plugin returned, with its message read out."""


def check_api_compatible(plugin: str, struct_size: int, major: int, minor: int) -> None:
    """Refuse a `PJRT_Api` table these bindings cannot read at fixed offsets.

    Same rule and wording as the JVM's `PjrtFfm.checkApiCompatible`.
    """
    if major != PJRT_API_MAJOR_SUPPORTED:
        raise PjrtError(
            f"PJRT plugin at {plugin} implements PJRT C API {major}.{minor}; Tlaloc binds "
            f"API major version {PJRT_API_MAJOR_SUPPORTED} and cannot call a plugin with a "
            f"different major version. Use a plugin built for PJRT C API {PJRT_API_MAJOR_SUPPORTED}.x."
        )
    if struct_size < PJRT_API_MIN_STRUCT_SIZE:
        raise PjrtError(
            f"PJRT plugin at {plugin} implements PJRT C API {major}.{minor} with a PJRT_Api table of "
            f"{struct_size} bytes; Tlaloc calls function pointers up to byte offset "
            f"{PJRT_API_HIGHEST_OFFSET_USED} and needs at least {PJRT_API_MIN_STRUCT_SIZE} bytes. "
            "The plugin is too old; use a newer one."
        )


# ---------------------------------------------------------------------------
# Finding a plugin file. Mirrors the JVM's `PjrtBinaries`: the same roots in
# the same order, every directory listing sorted, and a report of every place
# looked when nothing is found. Everything here is a FILE lookup; nothing is
# imported.
# ---------------------------------------------------------------------------

PLUGIN_PATH_ENV = "TLALOC_PJRT_PLUGIN_PATH"


def _sorted_children(path: str) -> list:
    try:
        return [os.path.join(path, n) for n in sorted(os.listdir(path))]
    except OSError:
        return []


def _search_roots(virtual_env: str | None, home: str | None) -> list:
    roots = []
    if virtual_env:
        roots.append(virtual_env)
    if home:
        roots.extend(p for p in _sorted_children(os.path.join(home, ".local", "venvs")) if os.path.isdir(p))
        roots.append(os.path.join(home, ".venv"))
        roots.append(os.path.join(home, "venv"))
        roots.append(os.path.join(home, ".local"))
    roots.append("/usr/local")
    roots.append("/usr")
    return roots


def _site_dirs(root: str) -> list:
    out = []
    for lib in ("lib", "lib64"):
        for py in _sorted_children(os.path.join(root, lib)):
            if os.path.basename(py).startswith("python3"):
                out.append(os.path.join(py, "site-packages"))
                out.append(os.path.join(py, "dist-packages"))
    return out


def _own_site_dirs() -> list:
    """This interpreter's own site-packages, searched after the fixed roots."""
    import site
    import sysconfig

    dirs = []
    for name in ("getsitepackages", "getusersitepackages"):
        fn = getattr(site, name, None)
        if fn is None:
            continue
        try:
            got = fn()
        except Exception:  # pragma: no cover - site is not always initialised
            continue
        dirs.extend([got] if isinstance(got, str) else list(got))
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        dirs.append(purelib)
    return dirs


def _gpu_family(platform: str) -> str | None:
    """The `jax_plugins/*<family>*` package name fragment for a GPU platform."""
    return {"cuda": "cuda", "gpu": "cuda", "rocm": "rocm"}.get(platform)


def plugin_candidates(platform: str, virtual_env: str | None, home: str | None,
                      extra_site_dirs: Sequence[str] = ()) -> list:
    """Every path a plugin for `platform` may live at, in resolution order.

    - cuda / gpu / rocm: `lib{,64}/python3.*/{site,dist}-packages/jax_plugins/*<family>*/*.so`
      under `$VIRTUAL_ENV`, `~/.local/venvs/*`, `~/.venv`, `~/venv`, `~/.local`,
      `/usr/local`, `/usr`, then under this interpreter's own site-packages.
    - tpu: `libtpu/libtpu.so` in `$VIRTUAL_ENV` and `~/.local` site-packages, then
      `/lib/libtpu.so` and `/usr/lib/libtpu.so`.
    - anything else (cpu): no standard install location; name the file explicitly.
    """
    if platform == "tpu":
        out = []
        for prefix in [p for p in (virtual_env, os.path.join(home, ".local") if home else None) if p]:
            for py in _sorted_children(os.path.join(prefix, "lib")):
                if os.path.basename(py).startswith("python3"):
                    out.append(os.path.join(py, "site-packages", "libtpu", "libtpu.so"))
        out += ["/lib/libtpu.so", "/usr/lib/libtpu.so"]
        return out
    family = _gpu_family(platform)
    if family is None:
        return []
    site_dirs = [d for root in _search_roots(virtual_env, home) for d in _site_dirs(root)]
    site_dirs += [d for d in extra_site_dirs if d not in site_dirs]
    out = []
    for site_dir in site_dirs:
        for pkg in _sorted_children(os.path.join(site_dir, "jax_plugins")):
            if family in os.path.basename(pkg).lower() and os.path.isdir(pkg):
                out.extend(f for f in _sorted_children(pkg) if f.endswith(".so"))
    return out


def _env_names_usable_plugin(platform: str, value: str) -> bool:
    # The TPU lane takes the generic variable only when it names a tpu-shaped
    # file, exactly as `PjrtBinaries.resolveTpuPlugin` does: on a CUDA host the
    # variable legitimately names xla_cuda_plugin.so.
    return platform != "tpu" or "tpu" in os.path.basename(value)


def find_plugin(platform: str = "cuda", explicit: str | None = None, *,
                env: dict | None = None, home: str | None = None,
                extra_site_dirs: Sequence[str] | None = None) -> str:
    """Locate a PJRT plugin `.so` for `platform`, or raise FileNotFoundError
    carrying the whole search report.

    Order: `explicit`, then `TLALOC_PJRT_PLUGIN_PATH`, then
    [plugin_candidates]. An explicit path or a set variable that names no file
    is refused rather than skipped.
    """
    env = os.environ if env is None else env
    home = os.path.expanduser("~") if home is None else home
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"no PJRT plugin at {explicit}")
        return explicit
    value = env.get(PLUGIN_PATH_ENV)
    if value and not os.path.exists(value):
        raise FileNotFoundError(f"{PLUGIN_PATH_ENV}={value} but no such file")
    extra = _own_site_dirs() if extra_site_dirs is None else extra_site_dirs
    found = _resolve(platform, env, home, extra)
    if found is None:
        raise FileNotFoundError(plugin_search_report(platform, env=env, home=home, extra_site_dirs=extra))
    return found


def _resolve(platform: str, env, home: str, extra: Sequence[str]) -> str | None:
    value = env.get(PLUGIN_PATH_ENV)
    if value and os.path.exists(value) and _env_names_usable_plugin(platform, value):
        return value
    for cand in plugin_candidates(platform, env.get("VIRTUAL_ENV"), home, extra):
        if os.path.exists(cand):
            return cand
    return None


def plugin_search_report(platform: str = "cuda", *, env: dict | None = None, home: str | None = None,
                         extra_site_dirs: Sequence[str] | None = None) -> str:
    """Every place [find_plugin] looks for `platform`, in order, and what is there."""
    env = os.environ if env is None else env
    home = os.path.expanduser("~") if home is None else home
    extra = _own_site_dirs() if extra_site_dirs is None else extra_site_dirs
    lines = [f"PJRT {platform} plugin resolution — where Tlaloc looked:"]
    value = env.get(PLUGIN_PATH_ENV)
    if not value:
        lines.append(f"  1. ${PLUGIN_PATH_ENV}: not set")
    elif not os.path.exists(value):
        lines.append(f"  1. ${PLUGIN_PATH_ENV} = {value} — no file there")
    elif not _env_names_usable_plugin(platform, value):
        lines.append(f"  1. ${PLUGIN_PATH_ENV} = {value} — not a tpu plugin name, ignored for tpu")
    else:
        lines.append(f"  1. ${PLUGIN_PATH_ENV} = {value} — FOUND")
    cands = plugin_candidates(platform, env.get("VIRTUAL_ENV"), home, extra)
    if platform == "tpu":
        lines.append("  2. libtpu locations, in order:")
        lines += [f"       {c} — {'FOUND' if os.path.exists(c) else 'missing'}" for c in cands]
    elif _gpu_family(platform) is None:
        lines.append(f"  2. no standard install location for a '{platform}' plugin; set ${PLUGIN_PATH_ENV}")
    elif not cands:
        fam = _gpu_family(platform)
        venv = env.get("VIRTUAL_ENV")
        lines.append(
            f"  2. no jax_plugins/*{fam}*/*.so under any searched root. Roots searched: "
            f"$VIRTUAL_ENV{' = ' + venv if venv else ' (not set)'}, ~/.local/venvs/*, ~/.venv, ~/venv, "
            f"~/.local, /usr/local, /usr, and this interpreter's site-packages — each at "
            f"lib{{,64}}/python3.*/{{site,dist}}-packages/jax_plugins/*{fam}*/*.so"
        )
    else:
        lines.append("  2. candidates found, in resolution order:")
        lines += [f"       {c}" for c in cands]
    found = _resolve(platform, env, home, extra)
    if found:
        lines.append(f"Resolved: {found}")
    else:
        lines.append(
            f"Fix: export {PLUGIN_PATH_ENV}=/path/to/the/plugin.so"
            + (", or `pip install jax[cuda12]` into a venv under ~/.local/venvs/ (or activate it)."
               if _gpu_family(platform) == "cuda" else "."))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Args structs. Each mirrors the C header; ctypes supplies the padding the
# Kotlin spells by hand. Sizes are asserted against the Kotlin's below.
# ---------------------------------------------------------------------------

_VOIDP = ctypes.c_void_p
_SIZET = ctypes.c_size_t


class PJRT_Error_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("error", _VOIDP)]


class PJRT_Error_Message_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("error", _VOIDP),
        ("message", _VOIDP), ("message_size", _SIZET),
    ]


class _PJRT_NamedValue_Union(ctypes.Union):
    _fields_ = [
        ("string_value", _VOIDP),
        ("int64_value", ctypes.c_int64),
        ("int64_array_value", _VOIDP),
        ("float_value", ctypes.c_float),
        ("bool_value", ctypes.c_bool),
    ]


class PJRT_NamedValue(ctypes.Structure):
    """§0.4.333 — the entry format for `PJRT_Client_Create_Args.create_options`.

    56 bytes: struct_size@0, extension_start@8, name@16, name_size@24,
    type@32 (+4 padding, because the union below is 8-aligned), value@40,
    value_size@48. `name` is NOT required to be NUL-terminated on the read
    side, but we terminate anyway — the Kotlin does.
    """

    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("name", _VOIDP), ("name_size", _SIZET),
        ("type", ctypes.c_int), ("value", _PJRT_NamedValue_Union),
        ("value_size", _SIZET),
    ]


class PJRT_Client_Create_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("create_options", _VOIDP), ("num_options", _SIZET),
        ("kv_get_callback", _VOIDP), ("kv_get_user_arg", _VOIDP),
        ("kv_put_callback", _VOIDP), ("kv_put_user_arg", _VOIDP),
        ("client", _VOIDP),
        ("kv_try_get_callback", _VOIDP), ("kv_try_get_user_arg", _VOIDP),
    ]


class PJRT_Client_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("client", _VOIDP)]


class PJRT_Client_PlatformName_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("client", _VOIDP),
        ("platform_name", _VOIDP), ("platform_name_size", _SIZET),
    ]


class PJRT_Client_AddressableDevices_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("client", _VOIDP),
        ("addressable_devices", _VOIDP), ("num_addressable_devices", _SIZET),
    ]


class PJRT_Program(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("code", _VOIDP), ("code_size", _SIZET),
        ("format", _VOIDP), ("format_size", _SIZET),
    ]


class PJRT_Client_Compile_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("client", _VOIDP),
        ("program", _VOIDP), ("compile_options", _VOIDP), ("compile_options_size", _SIZET),
        ("executable", _VOIDP),  # out
    ]


class PJRT_Client_BufferFromHostBuffer_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("client", _VOIDP),
        ("data", _VOIDP), ("type", ctypes.c_int),
        ("dims", _VOIDP), ("num_dims", _SIZET),
        ("byte_strides", _VOIDP), ("num_byte_strides", _SIZET),
        ("host_buffer_semantics", ctypes.c_int),
        ("device", _VOIDP), ("memory", _VOIDP), ("device_layout", _VOIDP),
        ("done_with_host_buffer", _VOIDP),  # out
        ("buffer", _VOIDP),                 # out
    ]


class PJRT_Buffer_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("buffer", _VOIDP)]


class PJRT_Buffer_OnDeviceSizeInBytes_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("buffer", _VOIDP),
        ("on_device_size_in_bytes", _SIZET),  # out
    ]


class PJRT_Buffer_ToHostBuffer_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("src", _VOIDP), ("host_layout", _VOIDP),
        ("dst", _VOIDP), ("dst_size", _SIZET),
        ("event", _VOIDP),  # out
    ]


class PJRT_LoadedExecutable_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("executable", _VOIDP)]


class PJRT_LoadedExecutable_GetExecutable_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("loaded_executable", _VOIDP), ("executable", _VOIDP),  # out
    ]


class PJRT_Executable_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("executable", _VOIDP)]


class PJRT_Executable_NumOutputs_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP), ("executable", _VOIDP),
        ("num_outputs", _SIZET),  # out
    ]


class PJRT_ExecuteOptions(ctypes.Structure):
    """All-zero except `struct_size` is the single-device, no-callback form.

    §0.4.459's audit applies unchanged: the fields a TPU would care about
    beyond CUDA are exactly the ones zeroed — `launch_id` (cross-host
    collective matching), `num_tasks` / `task_ids` / `incarnation_ids` /
    `multi_slice_config` (multi-host, multi-slice). Zero is the documented
    single-host single-task default. The 4 bytes of padding the Kotlin adds
    after the i32 `launch_id` are supplied here by ctypes' own alignment,
    and `_assert_layouts` proves the two agree (120 bytes).
    """

    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("send_callbacks", _VOIDP), ("recv_callbacks", _VOIDP),
        ("num_send_ops", _SIZET), ("num_recv_ops", _SIZET),
        ("launch_id", ctypes.c_int),
        ("non_donatable_input_indices", _VOIDP),
        ("num_non_donatable_input_indices", _SIZET),
        ("context", _VOIDP), ("call_location", _VOIDP),
        ("num_tasks", _SIZET), ("task_ids", _VOIDP),
        ("incarnation_ids", _VOIDP), ("multi_slice_config", _VOIDP),
    ]


class PJRT_LoadedExecutable_Execute_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", _SIZET), ("extension_start", _VOIDP),
        ("executable", _VOIDP), ("options", _VOIDP),
        ("argument_lists", _VOIDP), ("num_devices", _SIZET), ("num_args", _SIZET),
        ("output_lists", _VOIDP), ("device_complete_events", _VOIDP),
        ("execute_device", _VOIDP),
    ]


class PJRT_Event_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("event", _VOIDP)]


class PJRT_Event_Await_Args(ctypes.Structure):
    _fields_ = [("struct_size", _SIZET), ("extension_start", _VOIDP), ("event", _VOIDP)]


#: Sizes the Kotlin computes from its hand-written `MemoryLayout`s. The
#: JVM-side certification compares these numbers against `PjrtFfm.SZ_*`; a
#: disagreement means one of the two bindings is writing a field into the
#: wrong hole, which is the failure that produces a segfault three calls later
#: rather than an error at the call site.
LAYOUT_SIZES = {
    "PJRT_NamedValue": ctypes.sizeof(PJRT_NamedValue),
    "PJRT_Client_Create_Args": ctypes.sizeof(PJRT_Client_Create_Args),
    "PJRT_Client_Destroy_Args": ctypes.sizeof(PJRT_Client_Destroy_Args),
    "PJRT_Client_PlatformName_Args": ctypes.sizeof(PJRT_Client_PlatformName_Args),
    "PJRT_Client_AddressableDevices_Args": ctypes.sizeof(PJRT_Client_AddressableDevices_Args),
    "PJRT_Program": ctypes.sizeof(PJRT_Program),
    "PJRT_Client_Compile_Args": ctypes.sizeof(PJRT_Client_Compile_Args),
    "PJRT_Client_BufferFromHostBuffer_Args": ctypes.sizeof(PJRT_Client_BufferFromHostBuffer_Args),
    "PJRT_Buffer_Destroy_Args": ctypes.sizeof(PJRT_Buffer_Destroy_Args),
    "PJRT_Buffer_OnDeviceSizeInBytes_Args": ctypes.sizeof(PJRT_Buffer_OnDeviceSizeInBytes_Args),
    "PJRT_Buffer_ToHostBuffer_Args": ctypes.sizeof(PJRT_Buffer_ToHostBuffer_Args),
    "PJRT_LoadedExecutable_Destroy_Args": ctypes.sizeof(PJRT_LoadedExecutable_Destroy_Args),
    "PJRT_LoadedExecutable_GetExecutable_Args": ctypes.sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
    "PJRT_Executable_Destroy_Args": ctypes.sizeof(PJRT_Executable_Destroy_Args),
    "PJRT_Executable_NumOutputs_Args": ctypes.sizeof(PJRT_Executable_NumOutputs_Args),
    "PJRT_ExecuteOptions": ctypes.sizeof(PJRT_ExecuteOptions),
    "PJRT_LoadedExecutable_Execute_Args": ctypes.sizeof(PJRT_LoadedExecutable_Execute_Args),
    "PJRT_Event_Destroy_Args": ctypes.sizeof(PJRT_Event_Destroy_Args),
    "PJRT_Event_Await_Args": ctypes.sizeof(PJRT_Event_Await_Args),
}

#: Field offsets whose value is not obvious from the field order, because a
#: padding hole precedes them. These are the ones a hand-written binding gets
#: wrong, so they are reported and compared rather than trusted.
LAYOUT_OFFSETS = {
    "PJRT_NamedValue.type": PJRT_NamedValue.type.offset,
    "PJRT_NamedValue.value": PJRT_NamedValue.value.offset,
    "PJRT_NamedValue.value_size": PJRT_NamedValue.value_size.offset,
    "PJRT_ExecuteOptions.launch_id": PJRT_ExecuteOptions.launch_id.offset,
    "PJRT_ExecuteOptions.non_donatable_input_indices":
        PJRT_ExecuteOptions.non_donatable_input_indices.offset,
    "PJRT_Client_BufferFromHostBuffer_Args.type":
        PJRT_Client_BufferFromHostBuffer_Args.type.offset,
    "PJRT_Client_BufferFromHostBuffer_Args.dims":
        PJRT_Client_BufferFromHostBuffer_Args.dims.offset,
    "PJRT_Client_BufferFromHostBuffer_Args.host_buffer_semantics":
        PJRT_Client_BufferFromHostBuffer_Args.host_buffer_semantics.offset,
    "PJRT_Client_BufferFromHostBuffer_Args.device":
        PJRT_Client_BufferFromHostBuffer_Args.device.offset,
}


def _assert_layouts() -> None:
    """Fail at import, not at a segfault three calls later.

    Every number here is the Kotlin's, hand-computed from the header in
    §0.4.303/§0.4.304. If a future Python or platform changes ctypes'
    alignment rules, this is where it stops.
    """
    expected = {
        "PJRT_NamedValue": 56,
        "PJRT_Client_Create_Args": 88,
        "PJRT_Program": 48,
        "PJRT_Client_Compile_Args": 56,
        "PJRT_Client_BufferFromHostBuffer_Args": 120,
        "PJRT_Buffer_ToHostBuffer_Args": 56,
        "PJRT_ExecuteOptions": 120,
        "PJRT_LoadedExecutable_Execute_Args": 80,
    }
    for name, want in expected.items():
        got = LAYOUT_SIZES[name]
        if got != want:
            raise AssertionError(
                f"{name} is {got} bytes under this ctypes, but the certified Kotlin "
                f"binding (PjrtFfm.kt) lays it out as {want}. "
                "One of the two is writing fields into the wrong holes."
            )
    if ctypes.sizeof(_VOIDP) != 8:
        raise AssertionError("this binding assumes LP64 (aarch64 / x86_64); pointers are not 8 bytes here")


_assert_layouts()


# ---------------------------------------------------------------------------
# Client options — §0.4.333, the reboot incident, restated because it is the
# one mistake in this file that costs a machine rather than a test.
# ---------------------------------------------------------------------------


class PjrtClientOptions:
    """GPU-client allocator options passed as `create_options`.

    **Why this is not optional (2026-07-18, twice).** With zero create_options
    the CUDA plugin defaults to `preallocate=true` + `memory_fraction=0.75`:
    at client create it `cuMemAlloc`s 75% of "device memory" up front for the
    BFCAllocator. On the GB10 device memory *is* system RAM (128 GB unified,
    CPU-coherent), so every client pinned a ~98 GB unswappable pool. Stacking
    a few of those hard-hung the machine — the reboots that were blamed on
    kernels and were not.

    Defaults: **no preallocation** and a **0.5 fraction cap**, so even a
    runaway program leaves half the machine to the OS. Env overrides are the
    same names the JVM side reads, deliberately, so one box is configured
    once: `TLALOC_PJRT_MEMORY_FRACTION`, `TLALOC_PJRT_PREALLOCATE`.

    §0.4.459 — **these are the XLA GPU plugin's knobs, and they are gated by
    platform.** A TPU client is created with NO create_options at all
    (`create_options = NULL, num_options = 0`), because libtpu neither needs
    them nor is guaranteed to accept them, and an unknown NamedValue is the
    plugin's to reject. `for_platform()` below is that gate.
    """

    def __init__(self, memory_fraction: float = None, preallocate: bool = None):
        if memory_fraction is None:
            raw = os.environ.get("TLALOC_PJRT_MEMORY_FRACTION", "").strip()
            try:
                memory_fraction = float(raw) if raw else 0.5
            except ValueError:
                raise ValueError(
                    f"TLALOC_PJRT_MEMORY_FRACTION must be a number in (0, 1], got '{raw}'"
                ) from None
            if not (0.0 < memory_fraction <= 1.0):
                raise ValueError(
                    f"TLALOC_PJRT_MEMORY_FRACTION must be a number in (0, 1], got '{raw}'"
                )
        if preallocate is None:
            raw = os.environ.get("TLALOC_PJRT_PREALLOCATE", "").strip()
            if raw not in ("", "true", "false"):
                raise ValueError(f"TLALOC_PJRT_PREALLOCATE must be true or false, got '{raw}'")
            preallocate = raw == "true"
        if not (0.0 < memory_fraction <= 1.0):
            raise ValueError(f"memoryFraction must be in (0, 1], got {memory_fraction}")
        self.memory_fraction = memory_fraction
        self.preallocate = preallocate

    @staticmethod
    def for_platform(platform: str) -> "PjrtClientOptions | None":
        """§0.4.459's platform gate: CUDA gets the allocator options, anything
        else gets `None`, which marshals as "no options"."""
        return PjrtClientOptions() if platform in ("cuda", "gpu", "rocm") else None

    @property
    def named_value_count(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"PjrtClientOptions(memory_fraction={self.memory_fraction}, preallocate={self.preallocate})"


def marshal_create_options(options: PjrtClientOptions):
    """Build the `PJRT_NamedValue[]` for `create_options`.

    Returns `(array, keepalive)`; the caller must hold `keepalive` for as long
    as the plugin may read the array — the name buffers live in it. Mirrors
    `PjrtFfm.marshalCreateOptions`: entry 0 `memory_fraction` (kFloat), entry
    1 `preallocate` (kBool), `value_size = 1` for scalars per the header.

    The multi-node entries (`node_id` / `num_nodes`, §0.4.461) are NOT
    marshalled here. Not an oversight: on the JVM they exist so that a
    multi-node client can be *refused by name*, since the kv-store callbacks
    that such a client rendezvouses through are unimplemented (G4). A serving
    process has nothing to do with a second node until that lands, and a knob
    that can only be set to its refusal is worse than no knob.
    """
    count = options.named_value_count
    array = (PJRT_NamedValue * count)()
    keepalive = []

    def header(i: int, name: str, type_: int):
        name_bytes = name.encode("utf-8")
        buf = ctypes.create_string_buffer(name_bytes)
        keepalive.append(buf)
        array[i].struct_size = ctypes.sizeof(PJRT_NamedValue)
        array[i].name = ctypes.cast(buf, _VOIDP)
        array[i].name_size = len(name_bytes)
        array[i].type = type_
        array[i].value_size = 1

    header(0, "memory_fraction", PJRT_NAMED_VALUE_TYPE_FLOAT)
    array[0].value.float_value = options.memory_fraction
    header(1, "preallocate", PJRT_NAMED_VALUE_TYPE_BOOL)
    array[1].value.bool_value = options.preallocate
    keepalive.append(array)
    return array, keepalive


def normalise_mlir_for_xla(text: str) -> str:
    """Mirror of `PjrtApi.normaliseMlirForXla`.

    1. Rename the first `func.func @<sym>` to `@main` — XLA's compile pipeline
       requires the entry function to be named `main` ("conversion requires
       module with `main` function"), and Tlaloc emits `@<DxirFunction.name>`.
       Idempotent: a module that is already `@main` is unchanged.
    2. Wrap a bare `func.func` in a module, because `toStablehlo` produces
       function-only emits at the top level.
    """
    trimmed = text.strip()
    trimmed = re.sub(r"func\.func\s+@\w+", "func.func @main", trimmed, count=1)
    if not trimmed.startswith("module"):
        trimmed = "module @tlaloc_emit {\n" + trimmed + "\n}\n"
    return trimmed


# ---------------------------------------------------------------------------
# The handles. Lifetimes are context managers on purpose: every PJRT object
# here has an explicit Destroy, and a leaked client is a pinned GPU allocation
# that outlives the thing that made it.
# ---------------------------------------------------------------------------

_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
_FN_VOID = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class PjrtApi:
    """A loaded plugin's `PJRT_Api*`, with the function table bound."""

    def __init__(self, lib: ctypes.CDLL, api_ptr: int, plugin_path: str):
        self._lib = lib  # held so the library is not unloaded under us
        self._api = api_ptr
        self.plugin_path = plugin_path
        self._fn = {}

    @staticmethod
    def load(plugin_path: str = None, platform: str = "cuda") -> "PjrtApi":
        """dlopen a PJRT plugin and call its `GetPjrtApi`.

        `plugin_path` defaults to `TLALOC_PJRT_PLUGIN_PATH` — the same env var
        `PjrtBinaries` honours JVM-side, so one export configures both halves
        of the box. A plugin `.so` may live anywhere: inside a jax install
        (`jax_plugins/xla_cuda12/xla_cuda_plugin.so`), at `/lib/libtpu.so` on
        a TPU VM, or in a directory a deployment ships. Nothing here imports
        the package the file happens to sit in.

        With neither, [find_plugin] searches the same roots as the JVM's
        `PjrtBinaries` for a `platform` plugin. The loaded table's size and API
        version are checked before any function pointer is read.
        """
        path = find_plugin(platform, plugin_path)
        lib = ctypes.CDLL(path)
        try:
            get_api = lib.GetPjrtApi
        except AttributeError as exc:
            raise ValueError(
                f"the shared library at {path} does not export GetPjrtApi "
                "(not a PJRT-compatible plugin)"
            ) from exc
        get_api.restype = ctypes.c_void_p
        get_api.argtypes = []
        api_ptr = get_api()
        if not api_ptr:
            raise PjrtError(f"GetPjrtApi returned NULL for {path}")
        struct_size = ctypes.c_size_t.from_address(api_ptr + _OFF_API_STRUCT_SIZE).value
        major = ctypes.c_int32.from_address(api_ptr + _OFF_API_VERSION_MAJOR).value
        minor = ctypes.c_int32.from_address(api_ptr + _OFF_API_VERSION_MINOR).value
        check_api_compatible(path, struct_size, major, minor)
        api = PjrtApi(lib, api_ptr, path)
        api.api_version = (major, minor)
        api.api_struct_size = struct_size
        return api

    def _call(self, offset: int, args, void: bool = False):
        fn = self._fn.get(offset)
        if fn is None:
            raw = _VOIDP.from_address(self._api + offset).value
            if not raw:
                raise PjrtError(
                    f"PJRT_Api function pointer at offset {offset} is NULL — this plugin "
                    "does not implement the call (or the offset table is wrong for its API version)"
                )
            fn = (_FN_VOID if void else _FN)(raw)
            self._fn[offset] = fn
        return fn(ctypes.cast(ctypes.byref(args), _VOIDP))

    def _check(self, err_ptr) -> None:
        """Raise the plugin's message, then destroy the error. Always both."""
        if not err_ptr:
            return
        msg_args = PJRT_Error_Message_Args()
        msg_args.struct_size = ctypes.sizeof(PJRT_Error_Message_Args)
        msg_args.error = err_ptr
        self._call(OFFSET_PJRT_Error_Message, msg_args, void=True)
        message = _read_utf8(msg_args.message, msg_args.message_size)
        destroy = PJRT_Error_Destroy_Args()
        destroy.struct_size = ctypes.sizeof(PJRT_Error_Destroy_Args)
        destroy.error = err_ptr
        self._call(OFFSET_PJRT_Error_Destroy, destroy, void=True)
        raise PjrtError(message)

    def create_client(self, platform: str = "cuda", options="unset") -> "PjrtClient":
        """Create a PJRT client. **Never without create_options on CUDA.**

        `options="unset"` resolves through `PjrtClientOptions.for_platform`,
        the §0.4.459 gate. Passing `None` explicitly for a CUDA platform is
        refused by name: that is the §0.4.333 configuration that preallocates
        75% of unified memory and takes the machine down with it, and it is
        not something a caller should be able to ask for by accident.
        """
        if options == "unset":
            options = PjrtClientOptions.for_platform(platform)
        if options is None and platform in ("cuda", "gpu", "rocm"):
            raise ValueError(
                f"refusing to create a '{platform}' PJRT client with no create_options: "
                "the CUDA plugin then defaults to preallocate=true, memory_fraction=0.75, "
                "which on a unified-memory host (GB10) pins ~75% of system RAM per client "
                "and can hang the machine. Pass a "
                "PjrtClientOptions, or use a non-GPU platform where None is the correct form."
            )
        args = PJRT_Client_Create_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_Create_Args)
        keepalive = None
        if options is not None:
            array, keepalive = marshal_create_options(options)
            args.create_options = ctypes.cast(array, _VOIDP)
            args.num_options = options.named_value_count
        else:
            args.create_options = None
            args.num_options = 0
        self._check(self._call(OFFSET_PJRT_Client_Create, args))
        if not args.client:
            raise PjrtError("PJRT_Client_Create returned a null client without an error")
        del keepalive  # the plugin copies the options during the call
        return PjrtClient(self, args.client)


def _read_utf8(ptr, size) -> str:
    if not ptr or size <= 0:
        return ""
    return ctypes.string_at(ptr, size).decode("utf-8", errors="replace")


class PjrtClient:
    """A live `PJRT_Client*`. Use it as a context manager; forgetting to close
    leaks the device client and, on CUDA, its whole BFC pool."""

    def __init__(self, api: PjrtApi, ptr: int):
        self.api = api
        self._ptr = ptr

    def __enter__(self) -> "PjrtClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr is None:
            return
        args = PJRT_Client_Destroy_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_Destroy_Args)
        args.client = self._ptr
        self._ptr = None  # first, so a raise here cannot produce a double free
        self.api._check(self.api._call(OFFSET_PJRT_Client_Destroy, args))

    def _live(self) -> int:
        if self._ptr is None:
            raise PjrtError("this PJRT client has been destroyed")
        return self._ptr

    def platform_name(self) -> str:
        args = PJRT_Client_PlatformName_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_PlatformName_Args)
        args.client = self._live()
        self.api._check(self.api._call(OFFSET_PJRT_Client_PlatformName, args))
        return _read_utf8(args.platform_name, args.platform_name_size)

    def addressable_devices(self) -> list:
        args = PJRT_Client_AddressableDevices_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_AddressableDevices_Args)
        args.client = self._live()
        self.api._check(self.api._call(OFFSET_PJRT_Client_AddressableDevices, args))
        n = args.num_addressable_devices
        if n <= 0:
            raise PjrtError("PJRT_Client_AddressableDevices returned 0 devices")
        base = args.addressable_devices
        return [_VOIDP.from_address(base + i * 8).value for i in range(n)]

    def compile(self, stablehlo_mlir: str) -> "PjrtLoadedExecutable":
        """Compile StableHLO text. Format is `"mlir"`; the compile options are
        §0.4.304's six hand-encoded bytes, without which XLA Check-fails."""
        code = normalise_mlir_for_xla(stablehlo_mlir).encode("utf-8")
        code_buf = ctypes.create_string_buffer(code)
        fmt = b"mlir"
        fmt_buf = ctypes.create_string_buffer(fmt)
        opts_buf = ctypes.create_string_buffer(COMPILE_OPTIONS_PROTO_BYTES, len(COMPILE_OPTIONS_PROTO_BYTES))

        program = PJRT_Program()
        program.struct_size = ctypes.sizeof(PJRT_Program)
        program.code = ctypes.cast(code_buf, _VOIDP)
        program.code_size = len(code)
        program.format = ctypes.cast(fmt_buf, _VOIDP)
        program.format_size = len(fmt)

        args = PJRT_Client_Compile_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_Compile_Args)
        args.client = self._live()
        args.program = ctypes.cast(ctypes.byref(program), _VOIDP)
        args.compile_options = ctypes.cast(opts_buf, _VOIDP)
        args.compile_options_size = len(COMPILE_OPTIONS_PROTO_BYTES)
        self.api._check(self.api._call(OFFSET_PJRT_Client_Compile, args))
        return PjrtLoadedExecutable(self, args.executable)

    # --- host -> device -------------------------------------------------

    def _buffer_from_host(self, device, data_buf, nbytes: int, type_code: int, dims: Sequence[int]):
        dims_arr = (ctypes.c_int64 * len(dims))(*[int(d) for d in dims])
        args = PJRT_Client_BufferFromHostBuffer_Args()
        args.struct_size = ctypes.sizeof(PJRT_Client_BufferFromHostBuffer_Args)
        args.client = self._live()
        args.data = ctypes.cast(data_buf, _VOIDP)
        args.type = type_code
        args.dims = ctypes.cast(dims_arr, _VOIDP) if len(dims) else None
        args.num_dims = len(dims)
        args.host_buffer_semantics = HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL
        args.device = device
        # byte_strides / memory / device_layout stay NULL — dense row-major,
        # the device's default memory space, the layout the plugin picks.
        self.api._check(self.api._call(OFFSET_PJRT_Client_BufferFromHostBuffer, args))
        # kImmutableOnlyDuringCall means the copy is done when the call
        # returns, but the event still has to be freed (header contract).
        if args.done_with_host_buffer:
            self.api._destroy_event(args.done_with_host_buffer)
        return PjrtBuffer(self, args.buffer)

    def buffer_from_host_f32(self, device, values: Iterable[float], dims: Sequence[int]) -> "PjrtBuffer":
        vals = [float(v) for v in values]
        _check_count(len(vals), dims, "f32")
        return self._buffer_from_host(device, (ctypes.c_float * len(vals))(*vals), len(vals) * 4,
                                      PJRT_BUFFER_TYPE_F32, dims)

    def buffer_from_host_i32(self, device, values: Iterable[int], dims: Sequence[int]) -> "PjrtBuffer":
        vals = [int(v) for v in values]
        _check_count(len(vals), dims, "i32")
        return self._buffer_from_host(device, (ctypes.c_int32 * len(vals))(*vals), len(vals) * 4,
                                      PJRT_BUFFER_TYPE_S32, dims)

    def buffer_from_file(self, device, path, dtype: str, dims: Sequence[int]) -> "PjrtBuffer":
        """§0.4.480 — upload a RAW little-endian file straight to the device.

        The three `buffer_from_host_*` methods above take a Python iterable and
        materialise a `list` on the way. That is right for a block table and
        catastrophic for a weight: TinyLlama-1.1B staged as f32 is 1.1e9
        elements, and 1.1e9 Python floats is tens of gigabytes of PyObject
        before a single byte reaches the plugin.

        This path never creates a Python number. It allocates ONE ctypes buffer
        of exactly the file's length, `readinto`s it (no intermediate `bytes`
        object — `readinto` writes through the buffer protocol), and hands the
        address to PJRT. Peak host memory is one tensor, and the bytes are
        copied exactly once, by the plugin.

        The file is the operand byte for byte — dense row-major, little-endian,
        no header. That is `ServingArtifactWriter`'s staged-weight format, and
        the absence of a header is why this function can be four lines: the
        shape vocabulary lives in the manifest, not in the file.
        """
        try:
            type_code, width = RAW_DTYPES[dtype]
        except KeyError:
            raise ValueError(
                f"buffer_from_file: dtype '{dtype}' has no raw PJRT staging "
                f"(known: {sorted(RAW_DTYPES)})"
            ) from None
        n = 1
        for d in dims:
            n *= int(d)
        nbytes = n * width
        buf = (ctypes.c_char * nbytes)()
        with open(path, "rb") as f:
            got = f.readinto(buf)
        if got != nbytes:
            raise ValueError(
                f"buffer_from_file: {path} yielded {got} bytes but dims {list(dims)} "
                f"of {dtype} need {nbytes} — the artifact and the file disagree about "
                f"this operand's shape, which is a wrong answer that would run"
            )
        return self._buffer_from_host(device, buf, nbytes, type_code, dims)

    def buffer_from_host_bf16(self, device, patterns: Iterable[int], dims: Sequence[int]) -> "PjrtBuffer":
        """bf16 rides as raw `uint16` bit patterns, exactly as it does JVM-side
        (§0.4.455's convention, §0.4.457's PJRT lane): the upper 16 bits of an
        f32, moved verbatim. There is no numeric conversion anywhere on this
        path — widening is the caller's explicit act."""
        pats = [int(p) & 0xFFFF for p in patterns]
        _check_count(len(pats), dims, "bf16")
        return self._buffer_from_host(device, (ctypes.c_uint16 * len(pats))(*pats), len(pats) * 2,
                                      PJRT_BUFFER_TYPE_BF16, dims)


def _check_count(n: int, dims: Sequence[int], what: str) -> None:
    expected = 1
    for d in dims:
        expected *= int(d)
    if n != expected:
        raise ValueError(f"buffer_from_host_{what}: dims {list(dims)} imply {expected} elements, got {n}")


class PjrtBuffer:
    """A device buffer. Closing it is `PJRT_Buffer_Destroy`."""

    def __init__(self, client: PjrtClient, ptr: int):
        self.client = client
        self._ptr = ptr

    def __enter__(self) -> "PjrtBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr is None:
            return
        args = PJRT_Buffer_Destroy_Args()
        args.struct_size = ctypes.sizeof(PJRT_Buffer_Destroy_Args)
        args.buffer = self._ptr
        self._ptr = None
        self.client.api._check(self.client.api._call(OFFSET_PJRT_Buffer_Destroy, args))

    def _live(self) -> int:
        if self._ptr is None:
            raise PjrtError("this PJRT buffer has been destroyed")
        return self._ptr

    def device_size_in_bytes(self) -> int:
        args = PJRT_Buffer_OnDeviceSizeInBytes_Args()
        args.struct_size = ctypes.sizeof(PJRT_Buffer_OnDeviceSizeInBytes_Args)
        args.buffer = self._live()
        self.client.api._check(self.client.api._call(OFFSET_PJRT_Buffer_OnDeviceSizeInBytes, args))
        return args.on_device_size_in_bytes

    def _to_host_bytes(self, nbytes: int) -> bytes:
        dst = (ctypes.c_ubyte * nbytes)()
        args = PJRT_Buffer_ToHostBuffer_Args()
        args.struct_size = ctypes.sizeof(PJRT_Buffer_ToHostBuffer_Args)
        args.src = self._live()
        args.dst = ctypes.cast(dst, _VOIDP)
        args.dst_size = nbytes
        api = self.client.api
        api._check(api._call(OFFSET_PJRT_Buffer_ToHostBuffer, args))
        if args.event:  # the copy is asynchronous; wait, then free the event
            api._await_event(args.event)
            api._destroy_event(args.event)
        return bytes(dst)

    def to_f32(self, n: int) -> list:
        return list(struct.unpack(f"<{n}f", self._to_host_bytes(n * 4)))

    def to_f32_bits(self, n: int) -> list:
        """The f32 elements as raw bit patterns. The certification compares
        these rather than decimals: a bit pattern crosses the JSON boundary
        between this process and the JVM without a rounding story."""
        return list(struct.unpack(f"<{n}I", self._to_host_bytes(n * 4)))

    def to_i32(self, n: int) -> list:
        return list(struct.unpack(f"<{n}i", self._to_host_bytes(n * 4)))

    def to_bf16_patterns(self, n: int) -> list:
        return list(struct.unpack(f"<{n}H", self._to_host_bytes(n * 2)))


class PjrtLoadedExecutable:
    """A compiled `PJRT_LoadedExecutable*`, single-device execute only."""

    def __init__(self, client: PjrtClient, ptr: int):
        self.client = client
        self._ptr = ptr
        self._num_outputs = None

    def __enter__(self) -> "PjrtLoadedExecutable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr is None:
            return
        args = PJRT_LoadedExecutable_Destroy_Args()
        args.struct_size = ctypes.sizeof(PJRT_LoadedExecutable_Destroy_Args)
        args.executable = self._ptr
        self._ptr = None
        self.client.api._check(self.client.api._call(OFFSET_PJRT_LoadedExecutable_Destroy, args))

    def _live(self) -> int:
        if self._ptr is None:
            raise PjrtError("this PJRT loaded executable has been destroyed")
        return self._ptr

    def num_outputs(self) -> int:
        """`PJRT_Executable_NumOutputs` — reached through GetExecutable, whose
        returned handle is a separate object we must destroy again."""
        if self._num_outputs is not None:
            return self._num_outputs
        api = self.client.api
        get_args = PJRT_LoadedExecutable_GetExecutable_Args()
        get_args.struct_size = ctypes.sizeof(PJRT_LoadedExecutable_GetExecutable_Args)
        get_args.loaded_executable = self._live()
        api._check(api._call(OFFSET_PJRT_LoadedExecutable_GetExecutable, get_args))
        exe = get_args.executable
        try:
            num_args = PJRT_Executable_NumOutputs_Args()
            num_args.struct_size = ctypes.sizeof(PJRT_Executable_NumOutputs_Args)
            num_args.executable = exe
            api._check(api._call(OFFSET_PJRT_Executable_NumOutputs, num_args))
            self._num_outputs = int(num_args.num_outputs)
            return self._num_outputs
        finally:
            d = PJRT_Executable_Destroy_Args()
            d.struct_size = ctypes.sizeof(PJRT_Executable_Destroy_Args)
            d.executable = exe
            api._check(api._call(OFFSET_PJRT_Executable_Destroy, d))

    def execute(self, arg_buffers: Sequence[PjrtBuffer], device) -> list:
        """Single-device execute.

        `argument_lists` is `PJRT_Buffer* const* const*`: a one-element outer
        array (one device) pointing at the inner array of input buffers.
        `output_lists` has the same shape with the inner slots written by the
        plugin. The device-complete event is awaited and destroyed before
        returning, so a caller that reads an output buffer next sees the
        freshly computed values and not the previous contents.
        """
        api = self.client.api
        n_out = self.num_outputs()
        inner_args = (_VOIDP * max(len(arg_buffers), 1))(*[b._live() for b in arg_buffers])
        outer_args = (_VOIDP * 1)(ctypes.cast(inner_args, _VOIDP))
        inner_outputs = (_VOIDP * max(n_out, 1))()
        outer_outputs = (_VOIDP * 1)(ctypes.cast(inner_outputs, _VOIDP))
        device_complete = (_VOIDP * 1)()

        options = PJRT_ExecuteOptions()
        options.struct_size = ctypes.sizeof(PJRT_ExecuteOptions)
        # Everything else stays zero — see the PJRT_ExecuteOptions docstring.

        args = PJRT_LoadedExecutable_Execute_Args()
        args.struct_size = ctypes.sizeof(PJRT_LoadedExecutable_Execute_Args)
        args.executable = self._live()
        args.options = ctypes.cast(ctypes.byref(options), _VOIDP)
        args.argument_lists = ctypes.cast(outer_args, _VOIDP)
        args.num_devices = 1
        args.num_args = len(arg_buffers)
        args.output_lists = ctypes.cast(outer_outputs, _VOIDP)
        args.device_complete_events = ctypes.cast(device_complete, _VOIDP)
        args.execute_device = device

        api._check(api._call(OFFSET_PJRT_LoadedExecutable_Execute, args))
        if device_complete[0]:
            api._await_event(device_complete[0])
            api._destroy_event(device_complete[0])
        return [PjrtBuffer(self.client, inner_outputs[i]) for i in range(n_out)]


def _api_await_event(self: PjrtApi, event_ptr) -> None:
    args = PJRT_Event_Await_Args()
    args.struct_size = ctypes.sizeof(PJRT_Event_Await_Args)
    args.event = event_ptr
    self._check(self._call(OFFSET_PJRT_Event_Await, args))


def _api_destroy_event(self: PjrtApi, event_ptr) -> None:
    args = PJRT_Event_Destroy_Args()
    args.struct_size = ctypes.sizeof(PJRT_Event_Destroy_Args)
    args.event = event_ptr
    self._check(self._call(OFFSET_PJRT_Event_Destroy, args))


PjrtApi._await_event = _api_await_event
PjrtApi._destroy_event = _api_destroy_event


def f32_bits(value: float) -> int:
    """Host-side f32 bit pattern, for comparisons that must not go through a
    decimal. Mirrors `Float.toRawBits`."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def float_to_bf16_bits(value: float) -> int:
    """RNE narrowing, the §0.4.455 convention, in stdlib arithmetic.

    Round-to-nearest-even on the truncated 16 bits: add the rounding bias
    (0x7FFF plus the low bit of the surviving mantissa) before shifting.
    NaN is kept NaN with the quiet bit forced, because a NaN whose payload
    shifts into nothing becomes an infinity otherwise.
    """
    bits = f32_bits(value)
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return (rounded >> 16) & 0xFFFF


def bf16_bits_to_f32(pattern: int) -> float:
    return bits_to_f32((pattern & 0xFFFF) << 16)
