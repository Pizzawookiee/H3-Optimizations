"""Which prebuilt native binary belongs to this source revision.

Boring on purpose, and committed alongside the code it was built from. The
release tag names a source revision rather than a second version namespace,
because this is vendored CUDA for one pack, not a library anyone else
consumes.

The matrix is two rows. Nothing here needs to reason about torch versions,
CUDA wheel suffixes, Python versions or another package's ABI -- the ctypes
boundary removed all of that. Compare with sparse_install.py, which needed
every one of them to pick a Sparge wheel.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

# Bump when the native sources change. The tag of the GitHub Release holding
# the matching assets, and what a bug report should quote.
NATIVE_BUILD = 'native-v1'

# What h3_int8_abi_version() must return. A binary reporting anything else is
# refused rather than trusted, which is what makes a hand-copied stale file a
# clear error instead of a mystery.
REQUIRED_ABI = 1

_RELEASE_URL = (
    'https://github.com/Zironic/H3-Optimizations/releases/download/%s/%s'
)


@dataclass(frozen=True)
class NativeArtifact:
    filename: str
    sha256: str
    size: int

    @property
    def url(self):
        return _RELEASE_URL % (NATIVE_BUILD, self.filename)


# Filled in when the release assets are published. Until then the loader falls
# back to a local native/build/, which is how development works anyway.
ARTIFACTS: dict[tuple[str, str], NativeArtifact] = {
    # ('Windows', 'AMD64'): NativeArtifact(
    #     filename='h3_int8_attention-win-x64.dll',
    #     sha256='...',
    #     size=0,
    # ),
    # ('Linux', 'x86_64'): NativeArtifact(
    #     filename='libh3_int8_attention-linux-x64.so',
    #     sha256='...',
    #     size=0,
    # ),
}


def platform_key():
    return (platform.system(), platform.machine())


def artifact_for_this_platform():
    """The pinned artifact for this machine, or None if none is published."""
    return ARTIFACTS.get(platform_key())


def describe_platform():
    system, machine = platform_key()
    return '%s-%s' % (system.lower(), machine.lower())
