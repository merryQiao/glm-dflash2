# Model placeholder

Do not commit or automatically download GLM-5.2 here. On the Ascend server,
set `GLM52_MODEL_PATH` to the existing local checkpoint directory. The path
must contain `config.json`, tokenizer files and all local weight artifacts.
For Atlas 800 A2, follow the official multi-node GLM-5.2 capacity and
DP/TP/Expert-Parallel recipe; this placeholder does not imply single-node fit.
