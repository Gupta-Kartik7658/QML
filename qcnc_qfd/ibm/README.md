# IBM Quantum hardware runs

The hardware experiments use a **separate** Python environment, because `qiskit-ibm-runtime` needs Qiskit 2.x
and the experiment environment is pinned to Qiskit 1.3.1. The two environments exchange data only through files
in `results/<run_tag>/ibm_hardware/`.

```bash
python -m venv ibm-venv                      # Windows: ibm-venv\Scripts\activate
source ibm-venv/bin/activate
python -m pip install -r ibm/requirements-ibm.txt
```

**Save the IBM Quantum Platform account once.** Type the API key at a prompt so it never lands in a file of this
repository or in the shell history. It is stored in `~/.qiskit/qiskit-ibm.json`, outside the repository:

```python
from getpass import getpass
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=getpass("API key: "),
                                  instance="<your instance CRN>", set_as_default=True)
```

## Workflow

| Step | Environment | Command |
|---|---|---|
| 1. Build data bundles (rows, angles, ideal and simulated features, circuits as QPY) | experiment venv | `scripts/4_ibm_prepare.*` |
| 2. Dry run: compile for the device and print the QPU-time estimate | IBM venv | `scripts/5_ibm_submit.*` |
| 3. Submit the jobs | IBM venv | `scripts/5_ibm_submit.* --yes` |
| 4. Collect the results (jobs may queue for hours) | IBM venv | `python ibm/ibm_hardware.py fetch --all` |
| 5. Analyse | experiment venv | `scripts/6_ibm_analyze.*` |

## Design notes

* Several copies of the k-qubit feature-map circuit run side by side on disjoint physical qubits, so each shot
  measures several data rows. Row `r` of the bundle runs in execution `r // copies` on copy `r % copies`.
* No error mitigation, twirling or dynamical decoupling is used: the analysis measures the raw device behaviour.
* `--flatten` replaces parameter expressions such as `2*(pi - x_i)*(pi - x_j)` (ZZ map) by plain parameters whose
  values are computed client-side. Without it the ZZ jobs failed server-side with error 1502 (payload
  deserialization). A self-check verifies that the flattened circuit equals the original after binding.
* Each feature map is submitted as its own job (one PUB per job).
* The paper's seven jobs used 247 s of QPU time on `ibm_marrakesh` and `ibm_fez` (IBM Quantum Open Plan).
