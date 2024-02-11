import jax.numpy as np
from matplotlib import pyplot as plt
from jax.scipy.linalg import expm

#### Shift and Antidiagonal operators
def shift(size: int, unitary: bool = False):
    result = np.roll(np.eye(size), 1, axis=1)
    
    if not unitary:
        result = result.at[:, 0].set(np.zeros(size))    
    return result

def antidiagonal(size):
    result = np.zeros((size,size))
    for i in range(size):
        result = result.at[i,size-i-1].set(1)
    return result

def measure(st):
    assert st.shape[0] == 6*N # 6 because we have internal DOFs subspace of dim 6
    
    return (abs(st[:N])**2 + abs(st[N:2*N])**2 + abs(st[2*N:3*N])**2 + abs(st[3*N:4*N])**2 + abs(st[4*N:5*N])**2 + abs(st[5*N:])**2).reshape(-1, 1)

def O_inter(lamda):
    return np.array([[        0,            0,             -np.conjugate(lamda)],
                     [        0,            0,             -np.conjugate(lamda)],
                     [   -lamda,       -lamda,                    0           ]])

def assembly_operator(N,p,lamda,unitary = False):
    #### Operator P_
    P_ = np.kron( np.eye(6), np.eye(N) )
    P_ = P_  - np.kron( antidiagonal(6), antidiagonal(N) )
    P_ = 1/2*P_

    #### Operator D_p
    epi = np.exp(2j*p)
    epi_ = np.exp(-2j*p)
    D_p = np.kron(np.array([[epi, 0,  0, 0,  0,  0],
                            [ 0,  epi_,0, 0, 0,  0],
                            [ 0,   0,  0, 0, 0,  0],
                            [ 0,   0,  0,epi,0,  0],
                            [ 0,   0,  0, 0,epi_,0],
                            [ 0,   0,  0, 0, 0,  0] ]), np.eye(N) )

    I3 = np.array([[0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,1,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]])

    I6 = np.array([[0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,1]])

    #### Free evolution
    Tdaga2 = np.transpose(shift(N,unitary=unitary))
    Tdaga2 = Tdaga2@Tdaga2
    T2     = shift(N,unitary=unitary)
    T2     = T2@T2 
    D_p = D_p + np.kron(I3,Tdaga2) + np.kron(I6,T2)

    #### Operator J
    O = O_inter(lamda)
    zeroket = np.zeros((N,N))
    zeroket = zeroket.at[N//2,N//2].set(1) 
    J = -1j*np.kron(np.eye(2),O)
    J = np.kron(J,zeroket)
    J = expm(J)

    #### Assembly operator
    #A = np.kron(np.eye(6),T2) # test shift
    #A = P_ @ D_p @ P_ # test free evolution
    #A = J @ D_p @ P_
    #A = D_p
    A = J@D_p

    return A

def numerical_simulation(N,p,lamda,state,steps=10):
    
    # Numerical evaluation of initial state (Anm1 = A_{n - 1})
    Anm1 = state
    Anm1 = Anm1 / np.sqrt(sum(abs(Anm1)**2)) # Normalization
    
    result = measure(Anm1)
    walk   = assembly_operator(N,p,lamda) 
    for _ in range(steps):
        Anm1 = walk @ Anm1
        result = np.concatenate([result, measure(Anm1)], axis=1)

    return result

if __name__=='__main__':
    #### Parameters of the simulation
    # No. of sites:
    N = 50 + 1 # one spatial dim is |0X0|

    p     = np.pi/4
    lamda0 = np.pi/np.sqrt(2)

    #### STATES ####
    #### Internal DOFs
    # |0 > vector 
    zeropos = np.zeros(N)
    zeropos = zeropos.at[N//2].set(1.0)
    # Internal degrees of freedom basis
    e1 = np.array([1, 0, 0, 0, 0, 0])
    e2 = np.array([0, 1, 0, 0, 0, 0]) 
    e3 = np.array([0, 0, 1, 0, 0, 0]) 
    e4 = np.array([0, 0, 0, 1, 0, 0]) 
    e5 = np.array([0, 0, 0, 0, 1, 0]) 
    e6 = np.array([0, 0, 0, 0, 0, 1])

    #inpoot = np.array(np.kron((e1 + e2 + e3) / np.sqrt(3), zeropos),dtype=np.complex64)
    #inpoot = np.array(np.kron(e3, zeropos),dtype=np.complex64)
    #inpoot = np.array(np.kron((e1 + e2) / np.sqrt(2), zeropos),dtype=np.complex64)
    # \phi_{b-}
    # phibm = 1/2*( np.exp(-1j*p)*(e1-e4)+np.exp(1j*p)*(e2-e5) ) 
    # phibm = np.array(np.kron(phibm, zeropos),dtype=np.complex64)
    # \phi_{b+}
    phibp = 1/2*( np.exp(-1j*p)*(e1-e4)-np.exp(1j*p)*(e2-e5) )
    phibp = np.array(np.kron(phibp, zeropos),dtype=np.complex64)
    inpoot = phibp

    steps = 15

    results = []

    rows = 1
    cols = 4
    fig, ax = plt.subplots(nrows=rows, ncols = cols, figsize=(30, 8), gridspec_kw={'hspace': 0.5})

    #for i in range(rows):
    #    for j in range(cols):
    #        index = cols*i+j
    #        factor = 0.1
    #        lamda = lamda0 + factor*index*np.pi/np.sqrt(2)
    #        print(str(index)+") lambda value : "+str(lamda))
    #        results.append(numerical_simulation(N,p,lamda,inpoot,steps=steps))
#
    #        window: slice = slice(10, 40)
    #        ax[i][j].imshow(results[index][window, :20] / np.sum(results[index][window, :20], axis=0))
    #        formatted_value = "{:.1f}".format(factor*index)  # Format the value to have one decimal
    #        ax[i][j].set_title(r'$\lambda = \frac{\pi}{\sqrt{2}}$ + ' + formatted_value + r'$\frac{\pi}{\sqrt{2}}$', fontsize=12)
    #        ax[i][j].set_xlabel('Timestep')
    #        ax[i][j].set_ylabel('Relative position $y=x_1-x_2$')
    #        # Get current y-axis tick labels
    #        ticks = ax[i][j].get_yticks().tolist()
    #    
    #        # Shift the tick labels
    #        new_ticks = [str(int(tick - 15)) for tick in ticks]
    #        ax[i][j].set_yticklabels(new_ticks)

    for j in range(cols):
        index = j
        factor = 0.3
        lamda = lamda0 + factor*index*np.pi/np.sqrt(2)
        print(str(index)+") lambda value : "+str(lamda))
        results.append(numerical_simulation(N,p,lamda,inpoot,steps=steps))
    
        window: slice = slice(10, 40)
        im = ax[j].imshow(results[index][window, :20] / np.sum(results[index][window, :20], axis=0))
        formatted_value = "{:.1f}".format(factor*index)  # Format the value to have one decimal
        if factor*index != 0:
            ax[j].set_title(r'$\lambda = \frac{\pi}{\sqrt{2}}$ + ' + formatted_value + r'$\frac{\pi}{\sqrt{2}}$', fontsize=16, pad=20)
        else:
            ax[j].set_title(r'$\lambda = \frac{\pi}{\sqrt{2}}$', fontsize=16, pad=20)
            ax[j].set_ylabel('Relative position $y=x_1-x_2$', fontsize=14)
        ax[j].set_xlabel('Timestep', fontsize=14)
        # Get current y-axis tick labels
        ticks = ax[j].get_yticks().tolist()
    
        # Shift the tick labels
        new_ticks = [str(int(tick - 15)) for tick in ticks]
        ax[j].set_yticklabels(new_ticks)

        if j == cols - 1:  # Check if it's the rightmost plot
            fig.colorbar(im, ax=ax[j], shrink=0.8)  # Add colorbar only for the rightmost plot
    
    # Adjust layout to ensure equal sizes
    plt.show()