# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

# Exit immediately if anything goes wrong
set -e

# Run tests
python3 -m pytest -m "not (gpu or experiment)"