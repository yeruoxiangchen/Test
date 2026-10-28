# TRELLIS.2 data-toolkit compatibility adapter

The public `microsoft/TRELLIS.2` repository documents and imports dataset
modules such as `datasets.ObjaverseXL`, but does not include those modules in
its released `data_toolkit` directory.  The local adapter in this directory
implements that documented interface without changing the TRELLIS.2 checkout.

Its catalog locations and Objaverse-XL download convention follow the
ObjaverseXL adapter distributed by TencentARC/Pixal3D, which is licensed under
the MIT License (Copyright 2026 Tencent).  This implementation adds strict
frozen-UID filtering, SHA256 validation, safe single-member ZIP handling, and
the `no_file` callback required by the current TRELLIS.2 dual-grid script.

Both TRELLIS.2 and Pixal3D retain their own upstream licenses.  This adapter is
used only for dataset preparation; it is not a runtime dependency of the
pose-conditioned model.
