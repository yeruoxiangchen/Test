"""Second-generation ray-conditioned reconstruction; v1 checkpoints stay separate."""
import os

# The installed cumm implicit-GEMM NumPy bridge raises SIGFPE in this runtime.
# Select the already-supported native convolution before importing TRELLIS.
# This is process-local and does not modify third-party code or checkpoints.
os.environ.setdefault('SPCONV_ALGO', 'native')
