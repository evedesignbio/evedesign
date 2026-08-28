Vendored from https://github.com/OATML-Markslab/EVE at commit `460d70efeeeded58bc69227a203540d68953ae88`.

Copied verbatim:
- `EVE/EVE/{VAE_encoder,VAE_decoder,VAE_model}.py`, `EVE/EVE/default_model_params.json` -> `EVE/`
- `EVE/utils/data_utils.py` -> `utils/` (3 minimal compat fixes for numpy>=2/pandas>=2, no behavior change: `.shape` -> `.shape[0]` in two diagnostic print statements, and `msa_df.sequence[seq_idx]` -> `msa_df.sequence.iloc[seq_idx]` for positional indexing)

`utils/performance_helpers.py` and `utils/plot_helpers.py` (GMM calibration / plotting) were not vendored — out of scope.

See `LICENSE` (MIT) for the original license.
