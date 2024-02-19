from pathlib import Path

import pytest
import nbformat
import numpy as np
import matplotlib.pyplot as plt

basepath = Path(__file__).parent.parent / "docs" / "examples"


class TestExamples:
    @pytest.mark.parametrize("backend", ["sympy", "symengine"])
    @pytest.mark.parametrize(
        "notebook",
        filter(lambda x: x.suffix == ".ipynb", basepath.iterdir()),
        ids=lambda x: x.stem,
    )
    def test_notebook(self, notebook, backend):
        with open(notebook, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            # parse all code-cells from notebook
            code_cells = []
            for i in nb["cells"]:
                if (
                    i["cell_type"] == "code"
                    and "remove-input" not in i["metadata"]["tags"]
                ):
                    code_cells.append(i["source"])

            code = ""
            # Ingest the following lines at the top of the code so we can
            # run the notebook with both sympy and symengine backends.
            code += "from rt1_model import set_lambda_backend\n"
            code += f"set_lambda_backend('{backend}')\n"

            for c in code_cells:
                for l in c.split("\n"):
                    # strip off plt.show() to avoid blocking the terminal
                    l = l.replace("plt.show()", "")

                    # exclude lines that use IPython magic commands
                    # (e.g. starting with %)
                    if not l.startswith("%"):
                        code += f"{l}\n"

            # run code (use a shared dict for locals and globals to avoid issues
            # with undefined variables)
            d = dict()
            exec(code, d, d)

            # in case the notebook defines simulation and retrieval parameters,
            # named "found_params" and "sim_params", check for approx. equality
            if all(i in d for i in ("found_params", "sim_params")):
                for key, val in d["found_params"].items():
                    diff = np.abs(d["sim_params"][key] - val)

                    assert (
                        np.mean(diff) < 0.15
                    ), f"Fit results of parameter {key} are not within limits!"

        plt.close("all")
