"""Identity of the native binaries committed with this source revision."""

from __future__ import annotations

import platform

# Bump when the native sources change. The tag of the GitHub Release holding
# the matching assets, and what a bug report should quote.
NATIVE_BUILD = 'native-v1'

REQUIRED_ABI = 1


def platform_key():
    return (platform.system(), platform.machine())


def describe_platform():
    system, machine = platform_key()
    return '%s-%s' % (system.lower(), machine.lower())
