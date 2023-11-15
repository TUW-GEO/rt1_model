import nbformat
import pytest
from pathlib import Path
import numpy as np

basepath = Path(__file__).parent.parent / "docs" / "examples"


class TestExampleNotebooks:
    @pytest.mark.parametrize(
        "notebook", filter(lambda x: x.suffix == ".ipynb", basepath.iterdir())
    )
    @pytest.mark.parametrize("backend", ["sympy", "symengine"])
    def test_notebook_exec(self, notebook, backend):
        found_params, sim_params = dict(), dict()

        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            # parse all code-cells from notebook
            # - exclude lines that use magic commands (e.g. starting with %)
            code_cells = [i["source"] for i in nb["cells"] if i["cell_type"] == "code"]
            code = ""
            for c in code_cells:
                for l in c.split("\n"):
                    if not l.startswith("%"):
                        # test both sympy and symengine backend
                        if "RT1(" in l:
                            l.replace("RT1(", f"RT1(lambda_backend={backend}, ")

                        code += f"{l}\n"

            # run code (use a shared dict for locals and globals to avoid issues
            # with undefined variables)
            d = dict()
            exec(code, d, d)

            # in case the notebook defines simulation and retrieval parameters,
            # check for approx. equality
            if all(i in d for i in ("found_params", "sim_params")):
                for key, val in d["found_params"].items():
                    diff = np.abs(d["sim_params"][key] - val)

                    assert (
                        np.mean(diff) < 0.15
                    ), "Fit results of parameter {key} are not within limits!"


if __name__ == "__main__":
    pytest.main()
