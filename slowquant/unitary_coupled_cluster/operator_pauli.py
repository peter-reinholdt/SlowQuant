from __future__ import annotations

import copy
import functools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.base import (
    kronecker_product_cached,
    pauli_to_mat,
)


@functools.cache
def a_op_spin(idx: int, dagger: bool, num_spin_orbs: int, num_elec: int) -> dict[str, complex]:
    if idx % 2 == 0:
        return a_op(idx // 2, "alpha", dagger, num_spin_orbs, num_elec)
    else:
        return a_op((idx - 1) // 2, "beta", dagger, num_spin_orbs, num_elec)


@functools.cache
def a_op(
    spinless_idx: int, spin: str, dagger: bool, number_spin_orbitals: int, number_of_electrons: int
) -> dict[str, complex]:
    idx = 2 * spinless_idx
    if spin == "beta":
        idx += 1
    operators = {}
    op1 = ""
    op2 = ""
    fac1: complex = 1
    fac2: complex = 1
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                op1 += "X"
                fac1 *= 0.5
                op2 += "Y"
                fac2 *= -0.5j
            else:
                op1 += "X"
                fac1 *= 0.5
                op2 += "Y"
                fac2 *= 0.5j
        elif i < idx:
            op1 += "Z"
            op2 += "Z"
        else:
            op1 += "I"
            op2 += "I"
    operators[op1] = fac1
    operators[op2] = fac2
    return operators


def expectation_value(
    bra: StateVector,
    pauliop: PauliOperator,
    ket: StateVector,
    use_csr: int = 10,
    active_cache: dict[str, float] | None = None,
) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError("Bra and Ket does not have same number of inactive orbitals")
    if len(bra._active) != len(ket._active):
        raise ValueError("Bra and Ket does not have same number of active orbitals")
    total = 0.0
    for op, fac in pauliop.operators.items():
        if abs(fac) < 10**-12:
            continue
        tmp = 1.0
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(pauli_to_mat(op[i]), ket.ket_inactive[:, i]))
        for i in range(len(bra.bra_virtual)):
            op_idx = i + len(bra.bra_inactive) + len(bra._active_onvector)
            tmp *= np.matmul(bra.bra_virtual[i], np.matmul(pauli_to_mat(op[op_idx]), ket.ket_virtual[:, i]))
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        active_start = len(bra.bra_inactive)
        active_end = active_start + number_active_orbitals
        tmp_active = 1.0
        active_pauli_string = op[active_start:active_end]
        if number_active_orbitals != 0:
            do_calculation = True
            if active_cache is not None:
                if active_pauli_string in active_cache:
                    tmp_active *= active_cache[active_pauli_string]
                    do_calculation = False
            if do_calculation:
                if number_active_orbitals >= use_csr:
                    operator = copy.deepcopy(ket.ket_active_csr)
                else:
                    operator = copy.deepcopy(ket.ket_active)
                for pauli_mat_idx, pauli_mat_symbol in enumerate(active_pauli_string):
                    pauli_mat = pauli_to_mat(pauli_mat_symbol)
                    prior = pauli_mat_idx
                    after = number_active_orbitals - pauli_mat_idx - 1
                    if pauli_mat_symbol == "I":
                        continue
                    if number_active_orbitals >= use_csr:
                        operator = kronecker_product_cached(prior, after, pauli_mat_symbol, True).dot(
                            operator
                        )
                    else:
                        operator = np.matmul(
                            kronecker_product_cached(prior, after, pauli_mat_symbol, False),
                            operator,
                        )
                if number_active_orbitals >= use_csr:
                    tmp_active *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
                else:
                    tmp_active *= np.matmul(bra.bra_active, operator)
                if active_cache is not None:
                    active_cache[active_pauli_string] = tmp_active
        total += fac * tmp * tmp_active
    if abs(total.imag) > 10**-10:
        print(f"WARNING, imaginary value of {total.imag}")
    if active_cache is not None:
        return total.real, active_cache
    else:
        return total.real


class PauliOperator:
    def __init__(self, operator: dict[str, complex]) -> None:
        self.operators = operator
        self.screen_zero = True

    def __add__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] += fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = fac
        return PauliOperator(new_operators)

    def __sub__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] -= fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = -fac
        return PauliOperator(new_operators)

    def __mul__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators: dict[str, complex] = {}
        for op1, val1 in self.operators.items():
            for op2, val2 in pauliop.operators.items():
                new_op = ""
                fac: complex = val1 * val2
                for pauli1, pauli2 in zip(op1, op2):
                    if pauli1 == "I":
                        new_op += pauli2
                    elif pauli2 == "I":
                        new_op += pauli1
                    elif pauli1 == pauli2:
                        new_op += "I"
                    elif pauli1 == "X" and pauli2 == "Y":
                        new_op += "Z"
                        fac *= 1j
                    elif pauli1 == "X" and pauli2 == "Z":
                        new_op += "Y"
                        fac *= -1j
                    elif pauli1 == "Y" and pauli2 == "X":
                        new_op += "Z"
                        fac *= -1j
                    elif pauli1 == "Y" and pauli2 == "Z":
                        new_op += "X"
                        fac *= 1j
                    elif pauli1 == "Z" and pauli2 == "X":
                        new_op += "Y"
                        fac *= 1j
                    elif pauli1 == "Z" and pauli2 == "Y":
                        new_op += "X"
                        fac *= -1j
                if new_op in new_operators:
                    new_operators[new_op] += fac
                else:
                    new_operators[new_op] = fac
                if self.screen_zero:
                    if abs(new_operators[new_op]) < 10**-12:
                        del new_operators[new_op]
        return PauliOperator(new_operators)

    def __rmul__(self, number: float) -> PauliOperator:
        new_operators = self.operators.copy()
        for op in self.operators:
            new_operators[op] *= number
            if self.screen_zero:
                if abs(new_operators[op]) < 10**-12:
                    del new_operators[op]
        return PauliOperator(new_operators)

    @property
    def dagger(self) -> PauliOperator:
        new_operators = {}
        for op, fac in self.operators.items():
            new_operators[op] = np.conj(fac)
        return PauliOperator(new_operators)

    def eval_operators(self, state_vector: StateVector) -> dict[str, float]:
        op_values = {}
        for op in self.operators:
            op_values[op] = expectation_value(state_vector, PauliOperator({op: 1}), state_vector)
        return op_values

    def matrix_form(self, use_csr: int = 10, is_real: bool = False) -> np.ndarray | ss.csr_matrix:
        num_spin_orbs = len(list(self.operators.keys())[0])
        if num_spin_orbs >= use_csr:
            matrix_form = ss.identity(2**num_spin_orbs, dtype=complex) * 0.0
        else:
            matrix_form = np.identity(2**num_spin_orbs, dtype=complex) * 0.0
        for op, fac in self.operators.items():
            if abs(fac) < 10**-12:
                continue
            if num_spin_orbs >= use_csr:
                tmp = ss.identity(2**num_spin_orbs, dtype=complex)
            else:
                tmp = np.identity(2**num_spin_orbs, dtype=complex)
            for pauli_mat_idx, pauli_mat_symbol in enumerate(op):
                prior = pauli_mat_idx
                after = num_spin_orbs - pauli_mat_idx - 1
                if pauli_mat_symbol == "I":
                    continue
                if num_spin_orbs >= use_csr:
                    A = kronecker_product_cached(prior, after, pauli_mat_symbol, True)
                    tmp = A.dot(tmp)
                else:
                    tmp = np.matmul(kronecker_product_cached(prior, after, pauli_mat_symbol, False), tmp)
            matrix_form += fac * tmp
        if num_spin_orbs >= use_csr:
            if matrix_form.getformat() != "csr":
                matrix_form = ss.csr_matrix(matrix_form)
        if is_real:
            matrix_form = matrix_form.astype(float)
        return matrix_form


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> PauliOperator:
    E = PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec)) * PauliOperator(
        a_op(q, "alpha", False, num_spin_orbs, num_elec)
    )
    E += PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec)) * PauliOperator(
        a_op(q, "beta", False, num_spin_orbs, num_elec)
    )
    return E


def epqrs(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int) -> PauliOperator:
    if p == r and q == s:
        operator = 2 * (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
        )
    elif p == q == r:
        operator = (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", False, num_spin_orbs, num_elec))
        )
        operator += (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "beta", False, num_spin_orbs, num_elec))
        )
    elif p == r == s:
        operator = (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
        )
        operator += (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", False, num_spin_orbs, num_elec))
        )
    elif q == s:
        operator = (
            PauliOperator(a_op(r, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
        )
        operator += (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(r, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
        )
    else:
        operator = (
            PauliOperator(a_op(r, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
        )
        operator += (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(r, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "beta", False, num_spin_orbs, num_elec))
        )
        operator -= (
            PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(r, "beta", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "beta", False, num_spin_orbs, num_elec))
        )
        operator -= (
            PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(r, "alpha", True, num_spin_orbs, num_elec))
            * PauliOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
            * PauliOperator(a_op(s, "alpha", False, num_spin_orbs, num_elec))
        )
    return operator


def Eminuspq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> PauliOperator:
    return Epq(p, q, num_spin_orbs, num_elec) - Epq(q, p, num_spin_orbs, num_elec)


def Hamiltonian(
    h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int
) -> PauliOperator:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_spatial_orbs = num_spin_orbs // 2
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            if p == 0 and q == 0:
                H_expectation = h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
            else:
                H_expectation += h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            for r in range(num_spatial_orbs):
                for s in range(num_spatial_orbs):
                    H_expectation += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec)
    return H_expectation


def commutator(A: PauliOperator, B: PauliOperator) -> PauliOperator:
    return A * B - B * A


def Hamiltonian_energy_only(
    h: np.ndarray,
    g: np.ndarray,
    c_mo: np.ndarray,
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
) -> PauliOperator:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_inactive_spatial_orbs = num_inactive_spin_orbs // 2
    num_active_spatial_orbs = num_active_spin_orbs // 2
    num_spin_orbs = num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs
    # Inactive one-electron
    for i in range(num_inactive_spatial_orbs):
        if i == 0:
            H_expectation = h_mo[i, i] * Epq(i, i, num_spin_orbs, num_elec)
        else:
            H_expectation += h_mo[i, i] * Epq(i, i, num_spin_orbs, num_elec)
    # Active one-electron
    for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
        for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            if p == 0 and q == 0 and num_inactive_spatial_orbs == 0:
                H_expectation = h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
            else:
                H_expectation += h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
    # Inactive two-electron
    for i in range(num_inactive_spatial_orbs):
        for j in range(num_inactive_spatial_orbs):
            H_expectation += 1 / 2 * g_mo[i, i, j, j] * epqrs(i, i, j, j, num_spin_orbs, num_elec)
            if i != j:
                H_expectation += 1 / 2 * g_mo[j, i, i, j] * epqrs(j, i, i, j, num_spin_orbs, num_elec)
    # Inactive-Active two-electron
    for i in range(num_inactive_spatial_orbs):
        for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
                H_expectation += 1 / 2 * g_mo[i, i, p, q] * epqrs(i, i, p, q, num_spin_orbs, num_elec)
                H_expectation += 1 / 2 * g_mo[p, q, i, i] * epqrs(p, q, i, i, num_spin_orbs, num_elec)
                H_expectation += 1 / 2 * g_mo[p, i, i, q] * epqrs(p, i, i, q, num_spin_orbs, num_elec)
                H_expectation += 1 / 2 * g_mo[i, p, q, i] * epqrs(i, p, q, i, num_spin_orbs, num_elec)
    # Active two-electron
    for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
        for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            for r in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
                for s in range(
                    num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs
                ):
                    H_expectation += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec)
    return H_expectation
