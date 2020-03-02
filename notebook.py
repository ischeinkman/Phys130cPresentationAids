#%%
from qutip import *
import numpy as np
import random

#%% [markdown]
# ## Utilities
#%% [markdown]
# Builds a state of the form $(k0 |0> + k1 |1>) \otimes |0>^{num\_zeros}$
# %%
def build_initial(k0, k1, num_zeros):
    # Normalize the state
    kmag = np.sqrt(k0.conjugate() * k0+ k1.conjugate() * k1)
    nk0 = k0/kmag
    nk1 = k1/kmag
    # Construct the state as k0 [1, 0] + k1 [0, 1]
    output = nk0 * basis(2, 0) + nk1 * basis(2, 1)

    # Add in the extra zeroes at the end
    for cur in range(0, num_zeros):
        output = tensor(output, basis(2, 0))
    return output .unit()
#%% [markdown]
# Applies a *classical* 1 bit bitflip error to density matrix $rho$
# with probability $p$. 
#%%
def apply_bitflip(rho, p, bit):
    flip_matrix = []
    num_bits = int(np.log(max(rho.shape)) /np.log(2))
    for idx in range(0, num_bits):
        if idx == bit:
            flip_matrix.append(sigmax())
        else:
            flip_matrix.append(identity(2))
    flip_op = tensor(flip_matrix)
    return (1 - p) * rho + p * flip_op * rho * flip_op.dag()
# %% [markdown]
# Operator to apply a *quantum* 1 bit bitflip error to a single qbit. 
# %%
def noise(epsilon):
    return Qobj(np.array([
        [np.sqrt(1 - epsilon), np.sqrt(epsilon)], 
        [np.sqrt(epsilon), np.sqrt(1 - epsilon)], 
    ]))

#%% [markdown]
# Applies an operator equivalent to $\sum_{i\in states} |i>$ to density 
# matrix $rho$, while still preserving the dimension of $rho$. 
#%%
def project_states(rho, states):
    ret = rho.copy()
    for xidx in range(0, rho.shape[0]):
        for yidx in range(0, rho.shape[1]):
            if not xidx in states and not yidx in states:
                ret.data[xidx, yidx] = 0.0 
    ret = ret.unit()
    return ret 

# %%
max_error = 0.1
error = max_error
qb_errors = [error]

#%% [markdown]
# ## Circuit components

#%% [markdown]
# ### Setup 
# This setup circuit duplicates the qubit $|\psi>$ across 3 different qubits via
# entanglement, creating state $<0|\psi>|0 0 0 > + <1|\psi> |1 1 1 >$. 
# The bottom 2 qubits are the ancilla we will use to measure the syndrome. 
# %%
setup_circuit = QubitCircuit(5, reverse_states=False)
setup_circuit.add_gate('CNOT', controls=[0], targets=[1])
setup_circuit.add_gate('CNOT', controls=[0], targets=[2])

#%% [markdown]
# ### Syndrome measurement 
# This circuit measures the "syndrome" value of the circuit; the top qbit is set to 1
# when bit 0 $\ne$ bit 1, and the bottom is set to 1 when bit 0 $\ne$ bit 2. 
#%%
measure_circuit = QubitCircuit(5, reverse_states=False)
measure_circuit.add_state('S', targets=[3, 4])
measure_circuit.add_gate('CNOT', controls=[0], targets=[3])
measure_circuit.add_gate('CNOT', controls=[1], targets=[3])
measure_circuit.add_gate('CNOT', controls=[0], targets=[4])
measure_circuit.add_gate('CNOT', controls=[2], targets=[4])

#%% [markdown]
# ### Full 3-bit Error Correction Circuit
# %%
three_bit = QubitCircuit(5, reverse_states=False)
three_bit.add_state('S', targets=[3, 4], state_type='output')
three_bit.add_state('\psi', targets=[0], state_type='input')
three_bit.add_state('0', targets=[1, 2, 3, 4], state_type='input')
three_bit.add_circuit(setup_circuit)
three_bit.user_gates = {"Error" : noise}
three_bit.add_gate('Error', targets=[0], arg_value= qb_errors[0], arg_label="\epsilon")
three_bit.add_gate('Error', targets=[1], arg_value= qb_errors[0], arg_label="\epsilon")
three_bit.add_gate('Error', targets=[2], arg_value= qb_errors[0], arg_label="\epsilon")
three_bit.add_circuit(measure_circuit)
display(three_bit.png)

#%% [markdown]
# ## Example 
# Example application using $(2|0> + i|1>)/\sqrt{5}$ and an error probability of 0.1
#%%
inpt = build_initial(1, 0.5j,  4)
display("Input density matrix:")
display(ket2dm(inpt).ptrace([0]))
display("Density matrix for error with no correction:")
display(apply_bitflip(ket2dm(inpt), error, 0).ptrace([0]))
duped = gate_sequence_product(setup_circuit.propagators()) * inpt
dupeddm = ket2dm(duped)
err1 = apply_bitflip(dupeddm, error, 0)
err2 = apply_bitflip(err1, error, 1)
err3 = apply_bitflip(err2, error, 2)
#%%
measure_circuit_matrix = gate_sequence_product(measure_circuit.propagators())
measured = measure_circuit_matrix * err3 * measure_circuit_matrix.dag()
reduced = measured.ptrace([0, 1, 2])
syndrome = measured.ptrace([3, 4])

#%%
display("Density matrix for the 3 bits after all 3 errors:")
display(reduced)
no_flip_matrix = project_states(reduced, [0, 7])
no_flip_corr = syndrome[0, 0] * no_flip_matrix
#%%
flip_3_matrix = project_states(reduced, [1, 6])
flip_3_corr = syndrome[1, 1] * expand_oper(sigmax(), 3, 2) * flip_3_matrix * expand_oper(sigmax(), 3, 2).dag()
#%%
flip_2_matrix = project_states(reduced, [2, 5])
flip_2_corr = syndrome[2, 2] * expand_oper(sigmax(), 3, 1) * flip_2_matrix * expand_oper(sigmax(), 3, 1).dag()
#%%
flip_1_matrix = project_states(reduced, [3, 4])
flip_1_corr = syndrome[3, 3] * expand_oper(sigmax(), 3, 0) * flip_1_matrix * expand_oper(sigmax(), 3, 0).dag()
#%%
corrected = no_flip_corr + flip_1_corr + flip_2_corr + flip_3_corr
display("Density matrix after corrections:")
display(corrected.extract_states([0, 7]))

# %%
