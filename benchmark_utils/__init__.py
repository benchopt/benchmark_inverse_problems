# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax. To import external packages in this file, use a
# `safe_import_context` named "import_ctx", as follows:

from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    from .hugging_face_torch_dataset import HuggingFaceTorchDataset
    from .image_dataset import ImageDataset
    from .constants import constants
