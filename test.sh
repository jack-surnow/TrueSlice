# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice by the anonymous submission authors.

# Exit immediately if anything goes wrong
set -e

# Run tests
python3 -m pytest -m "not (gpu or experiment)"