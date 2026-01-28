import cudaq
import numpy as np


class NoiseModeler:
  
  def noise_modeler(self, noiseconfig: list):

    if len(noiseconfig) != 4:
      raise ValueError('Invalid noise configuration. Valid config in form of [1q prob (float), 2q prob (float), 1q channel (str), 2q channel (str)].')

    prob1q    = noiseconfig[0]
    prob2q    = noiseconfig[1]
    channel1q = noiseconfig[2]
    channel2q = noiseconfig[3]

    if prob1q != None and not isinstance(prob1q, float):
       raise ValueError(f"Invalid 1-qubit channel error probability: {prob1q}.")
    if prob2q != None and not isinstance(prob2q, float):
       raise ValueError(f"Invalid 2-qubit channel error probability: {prob2q}.")
    if channel1q not in ['depolarization', 'bitflip', 'phaseflip', None]:
       raise ValueError(f"Invalid 1-qubit noise channel: {channel1q}. Valid options: 'depolarization', 'bitflip', 'phaseflip' and None.")
    if channel2q not in ['depolarization', 'bitflip', 'phaseflip', None]:
       raise ValueError(f"Invalid 1-qubit noise channel: {channel2q}. Valid options: 'depolarization', 'bitflip', 'phaseflip' and None.")
    
    
    cudaq.set_random_seed(42)
    noise_model = cudaq.NoiseModel()
    ##########################################################################################
    # One-qubit Noise Channels
    ##########################################################################################
    if prob1q == None:
       cudaq.unset_noise()
       pass
    else:
      if channel1q == 'depolarization':
        noise_channel1q = cudaq.DepolarizationChannel(prob1q)
      elif channel1q == 'bitflip':
        noise_channel1q = cudaq.BitFlipChannel(prob1q)
      elif channel1q == 'phaseflip':
        noise_channel1q = cudaq.PhaseFlipChannel(prob1q)
      elif channel1q == None:
          pass
          
      if channel1q != None:  
          noisy_ops = ["h", "rx", "x", "z", "ry", "rz"] # "h", "rx", "x", "z", "ry", "rz"
          for op in noisy_ops:
              noise_model.add_all_qubit_channel(op, noise_channel1q)
      else:
          pass

    ##########################################################################################
    # Two-qubit Noise Channels
    ##########################################################################################
    if prob2q == None:
       cudaq.unset_noise()
       pass
    else:
      if channel2q == 'depolarization':
        kraus_channel = cudaq.Depolarization2(prob2q)
      ##########################################################################################
      elif channel2q == 'bitflip':
        I = np.array([[1, 0],
                      [0, 1]], dtype=np.complex128)

        X = np.array([[0, 1],
                      [1, 0]], dtype=np.complex128)

        I_I = np.kron(I, I)  # I ⊗ I
        I_X = np.kron(I, X)  # I ⊗ X
        X_I = np.kron(X, I)  # X ⊗ I
        X_X = np.kron(X, X)  # X ⊗ X

        a00 = np.sqrt((1.0 - prob2q)**2)
        a01 = np.sqrt(prob2q * (1.0 - prob2q))
        a10 = np.sqrt(prob2q * (1.0 - prob2q))
        a11 = np.sqrt(prob2q**2)

        K00 = a00 * I_I
        K01 = a01 * I_X
        K10 = a10 * X_I
        K11 = a11 * X_X
        
        kraus_channel = cudaq.KrausChannel([K00, K01, K10, K11])
      ##########################################################################################
      elif channel2q == 'phaseflip':
        I = np.array([[1, 0],
                      [0, 1]], dtype=np.complex128)
        Z = np.array([[1, 0],
                      [0, -1]], dtype=np.complex128)

        I_I = np.kron(I, I)  
        I_Z = np.kron(I, Z)  
        Z_I = np.kron(Z, I) 
        Z_Z = np.kron(Z, Z)

        a00 = np.sqrt((1.0 - prob2q) ** 2)
        a01 = np.sqrt(prob2q * (1.0 - prob2q))
        a10 = np.sqrt(prob2q * (1.0 - prob2q))
        a11 = np.sqrt(prob2q ** 2)

        K00 = a00 * I_I
        K01 = a01 * I_Z
        K10 = a10 * Z_I
        K11 = a11 * Z_Z

        kraus_channel = cudaq.KrausChannel([K00, K01, K10, K11]) 
      ##########################################################################################
      elif channel2q == None:
          pass
      if channel2q != None:     
          noise_model.add_all_qubit_channel("cry", kraus_channel)
          noise_model.add_all_qubit_channel("cr1", kraus_channel)
          noise_model.add_all_qubit_channel("cx", kraus_channel)
          noise_model.add_all_qubit_channel("crz", kraus_channel)
    
    return noise_model