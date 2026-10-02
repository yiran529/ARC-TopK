"""Run the 16-cell 1B BF16 extension of the Muon blocking/socket matrix."""

from pathlib import Path

import run_table_v_muon_blocking_matrix as matrix


matrix.ARTIFACTS = (
    matrix.ROOT / "output/CM106-CM121-table-v-muon-1b-bf16-blocking-matrix"
)
matrix.CONTROLLER_PATH = Path(__file__)
matrix.START_NUMBER = 106
matrix.DTYPE = "bfloat16"
matrix.MAX_LENGTH = 64
matrix.MODELS = (("1b", "llama_1b.json"),)


if __name__ == "__main__":
    matrix.main()
