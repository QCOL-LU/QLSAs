"""Tests for qlsas.executer.Executer."""

import numpy as np
import pytest
from types import SimpleNamespace
from pytket.circuit import Circuit as TketCircuit
from qnexus import QuantinuumConfig

from qlsas.executer import Executer
from qlsas.transpiler import Transpiler
from qlsas.ibm_options import (
    IBMExecutionOptions,
    apply_ibm_error_mitigation_options,
)
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transpiled_2x2_circuit(aer_backend):
    """Build and transpile a 2x2 HHL circuit for the AerSimulator."""
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="measure_x", num_qpe_qubits=3, eig_oracle="classical")
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    circ = hhl.build_circuit(A, b)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    return transpiler.optimize()


class _DummySamplerOptions:
    def __init__(self):
        self.dynamical_decoupling = SimpleNamespace(
            enable=False,
            sequence_type="XX",
            scheduling_method="alap",
        )
        self.twirling = SimpleNamespace(
            enable_gates=False,
            num_randomizations=None,
            shots_per_randomization=None,
        )


class _FakeJob:
    def job_id(self):
        return "fake-job-id"

    def status(self):
        return "DONE"

    def result(self):
        return ["fake-result"]


class _FakeSampler:
    last_instance = None

    def __init__(self, mode):
        self.mode = mode
        self.options = _DummySamplerOptions()
        self.run_calls = []
        type(self).last_instance = self

    def run(self, circuits, shots):
        self.run_calls.append((circuits, shots))
        return _FakeJob()


class _FakeIBMBackend:
    name = "fake_ibm_backend"


# ===================================================================
# Session lifecycle
# ===================================================================

class TestSessionLifecycle:

    def test_open_session_aer_is_noop(self, aer_backend):
        ex = Executer()
        ex.open_session(aer_backend, verbose=False)
        assert not ex.session_active

    def test_close_session_when_none_is_noop(self):
        ex = Executer()
        ex.close_session(verbose=False)
        assert not ex.session_active

    def test_double_open_is_idempotent(self, aer_backend):
        ex = Executer()
        ex.open_session(aer_backend, verbose=False)
        ex.open_session(aer_backend, verbose=False)
        assert not ex.session_active

    def test_context_manager_aer(self, aer_backend):
        ex = Executer()
        with ex.session(aer_backend, verbose=False):
            assert not ex.session_active
        assert not ex.session_active


# ===================================================================
# Execution
# ===================================================================

class TestExecution:

    def test_run_aer_returns_result(self, aer_backend):
        tc = _transpiled_2x2_circuit(aer_backend)
        ex = Executer()
        result = ex.run(tc, aer_backend, shots=100, verbose=False)
        assert result is not None

    def test_result_has_join_data(self, aer_backend):
        tc = _transpiled_2x2_circuit(aer_backend)
        ex = Executer()
        result = ex.run(tc, aer_backend, shots=100, verbose=False)
        joined = result.join_data(names=["ancilla_flag_result", "x_result"])
        counts = joined.get_counts()
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 100

    def test_run_qnexus_not_implemented(self):
        ex = Executer()
        backend = QuantinuumConfig(device_name="H1-1LE")
        tket_circ = TketCircuit(2)
        tket_circ.H(0)
        with pytest.raises(NotImplementedError):
            ex.run(tket_circ, backend, shots=10, verbose=False)

    def test_invalid_backend_type(self):
        ex = Executer()
        with pytest.raises(ValueError, match="backend type"):
            ex.run(None, {"bad": True}, shots=10, verbose=False)

    def test_run_qiskit_applies_ibm_options_for_ibm_backend(self, monkeypatch):
        monkeypatch.setattr("qlsas.executer.Sampler", _FakeSampler)
        monkeypatch.setattr("qlsas.executer.IBMBackend", _FakeIBMBackend)

        ibm_options = IBMExecutionOptions(
            enable_error_mitigation=True,
            enable_dynamical_decoupling=True,
            dd_sequence_type="XpXm",
            dd_scheduling_method="asap",
            enable_gate_twirling=True,
            twirling_num_randomizations=5,
            twirling_shots_per_randomization=16,
        )
        ex = Executer()
        backend = _FakeIBMBackend()

        result = ex.run_qiskit(object(), backend, shots=100, ibm_options=ibm_options, verbose=False)

        sampler = _FakeSampler.last_instance
        assert result == "fake-result"
        assert sampler is not None
        assert sampler.options.dynamical_decoupling.enable is True
        assert sampler.options.dynamical_decoupling.sequence_type == "XpXm"
        assert sampler.options.dynamical_decoupling.scheduling_method == "asap"
        assert sampler.options.twirling.enable_gates is True
        assert sampler.options.twirling.num_randomizations == 5
        assert sampler.options.twirling.shots_per_randomization == 16

    def test_run_qiskit_ignores_ibm_options_for_non_ibm_backend(self, monkeypatch, aer_backend):
        monkeypatch.setattr("qlsas.executer.Sampler", _FakeSampler)
        monkeypatch.setattr("qlsas.executer.IBMBackend", _FakeIBMBackend)

        ibm_options = IBMExecutionOptions(
            enable_error_mitigation=True,
            enable_dynamical_decoupling=True,
            enable_gate_twirling=True,
            twirling_num_randomizations=3,
        )
        ex = Executer()

        ex.run_qiskit(object(), aer_backend, shots=100, ibm_options=ibm_options, verbose=False)

        sampler = _FakeSampler.last_instance
        assert sampler is not None
        assert sampler.options.dynamical_decoupling.enable is False
        assert sampler.options.twirling.enable_gates is False


class TestIBMErrorMitigationOptions:

    def test_disabled_mitigation_leaves_sampler_options_unchanged(self):
        sampler_options = _DummySamplerOptions()
        ibm_options = IBMExecutionOptions(
            enable_dynamical_decoupling=True,
            enable_gate_twirling=True,
            twirling_num_randomizations=4,
        )

        apply_ibm_error_mitigation_options(sampler_options, ibm_options)

        assert sampler_options.dynamical_decoupling.enable is False
        assert sampler_options.dynamical_decoupling.sequence_type == "XX"
        assert sampler_options.twirling.enable_gates is False
        assert sampler_options.twirling.num_randomizations is None

    def test_apply_error_mitigation_options_updates_sampler_options(self):
        sampler_options = _DummySamplerOptions()
        ibm_options = IBMExecutionOptions(
            enable_error_mitigation=True,
            enable_dynamical_decoupling=True,
            dd_sequence_type="XY4",
            dd_scheduling_method="alap",
            enable_gate_twirling=True,
            twirling_num_randomizations=6,
            twirling_shots_per_randomization=24,
        )

        apply_ibm_error_mitigation_options(sampler_options, ibm_options)

        assert sampler_options.dynamical_decoupling.enable is True
        assert sampler_options.dynamical_decoupling.sequence_type == "XY4"
        assert sampler_options.dynamical_decoupling.scheduling_method == "alap"
        assert sampler_options.twirling.enable_gates is True
        assert sampler_options.twirling.num_randomizations == 6
        assert sampler_options.twirling.shots_per_randomization == 24

    def test_invalid_twirling_num_randomizations_raises(self):
        with pytest.raises(ValueError, match="twirling_num_randomizations"):
            IBMExecutionOptions(twirling_num_randomizations=0)
