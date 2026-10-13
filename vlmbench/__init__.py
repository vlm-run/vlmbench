"""vlmbench — VLM benchmark CLI.

The eval catalog types are re-exported here so a catalog or run file needs one
import: `from vlmbench import Eval, Model, Prompt, Provider, Run, Sample`.
"""

from vlmbench.eval import Eval, EvalError, Model, Prompt, Provider, Run, Sample, Verdict
from vlmbench.version import __version__

__all__ = ["Eval", "EvalError", "Model", "Prompt", "Provider", "Run", "Sample", "Verdict", "__version__"]
