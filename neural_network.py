import numpy as np
from typing import Callable, List, Tuple, Union, Dict, Any
from string import ascii_letters


import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Columnwise softmax activation.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n, m), where n is the number of classes 
        and m is the number of examples.

    Returns
    -------
    np.ndarray
        Softmax applied columnwise, same shape as input.
    """
    # subtract max per column for numerical stability
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid wrt input x"""
    s = sigmoid(x)
    return s * (1 - s)

def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """Jacobian of softmax wrt input z (vector version)."""
    s = softmax(z).reshape(-1, 1)   # column vector
    return np.diagflat(s) - s @ s.T

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU wrt input x"""
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh wrt input x"""
    return 1 - np.tanh(x) ** 2


class ActivationFunctions:
    """
    Collection of activation functions and their derivatives.
    Provides a clean API to fetch function/derivative pairs.
    """
    functions: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh,
        'softmax': softmax,
        'identity': None
    }

    derivatives: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        'sigmoid': sigmoid_derivative,
        'relu': relu_derivative,
        'tanh': tanh_derivative,
        'softmax': softmax_derivative,
        'identity': None
    }

    @classmethod
    def get(cls, name: str) -> Callable[[np.ndarray], np.ndarray]:
        """Get activation function by name"""
        if name not in cls.functions:
            raise ValueError(f"Unknown activation function: {name}")
        return cls.functions[name]

    @classmethod
    def get_derivative(cls, name: str) -> Callable[[np.ndarray], np.ndarray]:
        """Get derivative function by name"""
        if name not in cls.derivatives:
            raise ValueError(f"Unknown activation function: {name}")
        return cls.derivatives[name]
    
class TensorJacobian(np.ndarray):
    """
    A subclass of numpy.ndarray that stores additional information
    useful for tensor derivatives.

    Attributes
    ----------
    tensor_type_numerator : List[int]
        The (r,s) type of the numerator tensor
    tensor_type_denominator : List[int]
        The (r,s) type of the denominator tensor
    """

    tensor_type_numerator : List[int]
    tensor_type_denominator : List[int]

    def __new__(cls, DTDS , tensor_type_numerator : List[int] , tensor_type_denominator : List[int]) -> "TensorJacobian":
        """
        Create a new TensorJacobian object from an input tensor.

        Parameters
        ----------
        DTDS : 
            The full tensor of DT/DS.

        Returns
        -------
        TensorJacobian
            An instance of TensorJacobian.
        """
        obj = DTDS.view(cls)

        obj.tensor_type_numerator = tensor_type_numerator
        obj.tensor_type_denominator = tensor_type_denominator
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """
        Finalize creation of the array, ensuring attributes are preserved
        when new views or slices are created.

        Parameters
        ----------
        obj : Any
            The object being viewed or sliced from.
        """
        if obj is None:
            return
        self.tensor_type_numerator = getattr(obj, 'tensor_type_numerator', None)
        self.tensor_type_denominator = getattr(obj, 'tensor_type_denominator', None)
    def __mul__(self, obj2):
        if not isinstance(obj2, TensorJacobian):
            return NotImplemented
        tensor_type_denominator1 = self.tensor_type_denominator
        tensor_type_numerator2 = obj2.tensor_type_numerator
        if tensor_type_denominator1  != tensor_type_numerator2:
            raise ValueError('Tensor dimensions do not align')
        einsum_string = _get_einsum_string_for_mul_tensor_jacobian(self.tensor_type_numerator, tensor_type_denominator1, tensor_type_numerator2 , obj2.tensor_type_denominator)
        try:
            result = np.einsum(einsum_string, self, obj2)
        except ValueError as e:
            print("EINSUM DEBUG")
            print("  einsum:", einsum_string)
            print("  self.shape:", self.shape, "types", self.tensor_type_numerator, self.tensor_type_denominator)
            print("  obj2.shape:", obj2.shape, "types", obj2.tensor_type_numerator, obj2.tensor_type_denominator)
            raise
        return TensorJacobian(result, self.tensor_type_numerator, obj2.tensor_type_denominator)

def _get_einsum_string_for_mul_tensor_jacobian(tensor_type_numerator1 : List[int] , tensor_type_denominator1 : List[int], tensor_type_numerator2 : List[int] , tensor_type_denominator2 : List[int]):
    '''
    Helper function for einsum string
    Example: for multiplication of two (1+1, 1+1) tensors, it will return the string abcd,dcef->abef
    '''

    tensor_dim_numerator1 = sum(tensor_type_numerator1)
    tensor_dim_denominator1 = sum(tensor_type_denominator1)
    tensor_dim_numerator2 = sum(tensor_type_numerator2)
    tensor_dim_denominator2 = sum(tensor_type_denominator2)
    letters = ascii_letters
    left_string = letters[0:tensor_dim_numerator1]
    letters = letters[tensor_dim_numerator1:]
    mid_string = letters[0:tensor_dim_denominator1]
    mid_string1 = mid_string[:tensor_type_denominator1[0]]
    mid_string2 = mid_string[tensor_type_denominator1[0]:]
    mid_string_swap = mid_string2 + mid_string1
    letters = letters[tensor_dim_denominator1:]
    right_string = letters[0:tensor_dim_denominator2]
    return left_string + mid_string + ',' + mid_string_swap +  right_string + '->' + left_string + right_string

def init_weights(layer_node_nums: List[int], radius: float = 1):

    layers = len(layer_node_nums)
    Ws = []
    bs = []
    for layer, layer_node_num in enumerate(layer_node_nums):
        if layer == 0:
            layer_node_num_previous = layer_node_num
            continue
        W = (np.random.rand(layer_node_num, layer_node_num_previous)-0.5)*radius
        b = (np.random.rand(layer_node_num,1)-0.5)*radius
        Ws.append(W)
        bs.append(b)
        layer_node_num_previous = layer_node_num
        
    return Ws, bs

def forward_propagate(X, Ws: List[np.ndarray], bs: List[np.ndarray], final_activation_function = 'softmax', activation_function = 'sigmoid'):
    layers = len(Ws)
    if final_activation_function == 'softmax':
        if Ws[-1].shape[0] == 1:
            print('Warning: softmax does not work when number of nodes is 1. Swapping to sigmoid.')
            final_activation_function = 'sigmoid'

    if layers != len(bs):
        raise ValueError('Ws and Bs need to be of the same length; the number of layers excluding the input.')
    As = []
    Zs = []
    previous_A = X
    act = ActivationFunctions.get(activation_function)
    for layer in range(layers):

        Wi = Ws[layer]
        bi = bs[layer]
        # Bi unneeded due to how numpy handles addition
        Zi = Wi.dot(previous_A) + bi
        if layer == layers - 1:
            act = ActivationFunctions.get(final_activation_function)
        if act:
            Ai = act(Zi)
        else:
            Ai = Zi
        previous_A = Ai
        As.append(Ai)
        Zs.append(Zi)
    return As, Zs

def columnwise_mse(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """
    Compute the error function:
    E(Y, Yhat) = (1 / (2m)) * sum(||y_i - yhat_i||^2) over columns
    
    Parameters
    ----------
    Y : np.ndarray
        Predicted values, shape (n, m)
    Yhat : np.ndarray
        True values, shape (n, m)
    
    Returns
    -------
    float
        Columnwise mean squared error divided by 2
    """
    if Y.shape != Yhat.shape:
        raise ValueError("Y and Yhat must have the same shape")

    m = Y.shape[1]  # number of columns
    diff = Y - Yhat
    squared_norms = np.sum(diff**2, axis=0)  # sum along rows (n)
    error = np.sum(squared_norms) / (2 * m)
    return error

def get_DEDY(Y,Yhat):
    if Y.shape != Yhat.shape:
        raise ValueError("Y and Yhat must have the same shape")
    m = Y.shape[1]  # number of columns
    DEDY = (1/m)*(Y - Yhat).T
    return TensorJacobian(DEDY, [0,0], [1,1])

def get_DSDZL(Z, activation_function = 'softmax'):
    if activation_function != 'softmax':
        return get_DSDZ(Z,activation_function=activation_function)
    else:
        nZ = Z.shape[0]
        mZ = Z.shape[1]
        nS = nZ
        mS = mZ
        DSsDZ = np.zeros((nS, mS, mZ, nZ))
        act = ActivationFunctions.get(activation_function)
        for i in range(nS):
            for j in range(mS):
                for l in range(mZ):
                    for k in range(nZ):
                        if l==j:
                            if i==k:
                                DSsDZ[i][j][l][k] = act(Z)[i,l]*(1 - act(Z)[k,l])

                            else:
                                DSsDZ[i][j][l][k] = -act(Z)[i,l]*act(Z)[k,l]
        return TensorJacobian(DSsDZ, [1,1], [1,1])

def get_DSDZ(Z, activation_function = 'sigmoid'):
    nZ = Z.shape[0]
    mZ = Z.shape[1]
    nS = nZ
    mS = mZ
    DSDZ = np.zeros((nS, mS, mZ, nZ))
    act = ActivationFunctions.get_derivative(activation_function)
    for i in range(nS):
        for j in range(mS):
            for l in range(mZ):
                for k in range(nZ):
                    if i==k and l==j:
                        DSDZ[i][j][l][k] = act(Z[k][l])
    return TensorJacobian(DSDZ, [1,1], [1,1])

def get_DZDW(W, A):
    '''
    Gets the derivative of Z = WA wrt W as a tensor
    '''
    nW = W.shape[0]
    mW = W.shape[1]
    nZ = nW
    mZ = A.shape[1]
    DZDW = np.zeros((nZ, mZ, mW, nW))
    for i in range(nZ):
        for j in range(mZ):
            for l in range(mW):
                for k in range(nW):
                    if i==k:
                        DZDW[i][j][l][k] = A[l][j]
    return TensorJacobian(DZDW, [1,1], [1,1])

def get_DZDA(W, A):
    '''
    Gets the derivative of Z = WA wrt A as a tensor
    '''
    nA = A.shape[0]
    mA = A.shape[1]
    nZ = W.shape[0]
    mZ = mA
    DZDA = np.zeros((nZ, mZ, mA, nA))
    for i in range(nZ):
        for j in range(mZ):
            for l in range(mA):
                for k in range(nA):
                    if l==j:
                        DZDA[i][j][l][k] = W[i][k]
    return TensorJacobian(DZDA, [1,1], [1,1])

def get_DZDb(b, Z):
    nZ = Z.shape[0]
    mZ = Z.shape[1]
    nb = b.shape[0]
    DZDb = np.zeros((nZ, mZ, nb))
    for i in range(nZ):
        for j in range(mZ):
            for k in range(nb):
                    if i==k:
                        DZDb[i][j][k] = 1
    return TensorJacobian(DZDb, [1,1], [1,0])

def back_propagate(X, Yhat, As, Zs, Ws, bs, final_activation_function = 'softmax', activation_function = 'sigmoid'):
    layers = len(As)
    if final_activation_function == 'softmax':
        if As[-1].shape[1] == 1:
            print('Warning: softmax does not work when number of nodes is 1. Swapping to sigmoid.')
            final_activation_function = 'sigmoid'
    if layers != len(bs):
        raise ValueError('Bad size As and bs')
    DEDWs = []
    DEDbs = []

    Y = As[-1]
    ZL = Zs[-1]
    WL = Ws[-1]
    ALminus1 = As[-2]
    bL = bs[-1]
    DEDY = get_DEDY(Y, Yhat)
    if final_activation_function != 'identity':
        DSLDZL = get_DSDZL(ZL, activation_function = final_activation_function)
    DZLDWL = get_DZDW(WL,ALminus1)
    DZLDbL = get_DZDb(bL,ZL)

    if final_activation_function == 'identity':
        DEDZL = DEDY
    else:
        DEDZL = DEDY * DSLDZL

    DEDWL = DEDZL * DZLDWL
    DEDbL = DEDZL * DZLDbL

    DEDWL = DEDWL.T
    DEDbL = DEDbL.T

    DEDWs.append(DEDWL)
    DEDbs.append(DEDbL)

    DEDZi = DEDZL
    for layer in range(layers - 1):

        Zi = Zs[-layer-2]
        Wi = Ws[-layer-2]
        Ai = As[-layer-2]
        Wiplus1 = Ws[-layer-1]
        if layer == layers - 2:
            Aiminus1 = X
        else:
            Aiminus1 = As[-layer-3]
        bi = bs[-layer-2]

        DZiplus1DAi = get_DZDA(Wiplus1,Ai)
        DSiDZi = get_DSDZ(Zi, activation_function = activation_function)

        DEmid = DZiplus1DAi * DSiDZi

        DZiDWi = get_DZDW(Wi,Aiminus1)
        DZiDbi = get_DZDb(bi,Zi)

        DEDZi = DEDZi * DEmid

        DEDWi = DEDZi * DZiDWi
        DEDbi = DEDZi * DZiDbi

        DEDWi = DEDWi.T
        DEDbi = DEDbi.T

        DEDWs.append(DEDWi)
        DEDbs.append(DEDbi)
    DEDWs.reverse()
    DEDbs.reverse()

    return DEDWs, DEDbs

    
if __name__ == "__main__":
    a = np.arange(2*4*5).reshape(2,4,5)
    print(a)
    b = np.arange(4*5*9).reshape(5,4,3,3)
    print(_get_einsum_string_for_mul_tensor_jacobian([1,0],[1,3],[1,3],[1,1]))
    c = np.einsum(_get_einsum_string_for_mul_tensor_jacobian([1,0],[1,1],[1,1],[1,1]),a,b)
    print(c)
    print(c.shape)
    print(ascii_letters)
    print (a.reshape(1,-1))
    print(_get_einsum_string_for_mul_tensor_jacobian([1,0],[1,1],[1,1],[1,1]))
    Ta = TensorJacobian(a,[1,0],[1,1])
    Tb = TensorJacobian(b,[1,1],[1,1])
    print(Ta*Tb)

    A = np.arange(12).reshape(3,4)
    B = np.arange(20).reshape(4,5)
    print(A.dot(B))
    print('nice')

    layer_node_nums = [5,2,3,2]
    Ws, bs = init_weights([5,2,3,2],1)
    for i, W in enumerate(Ws):
        # print(W)
        # print(bs[i])
        print(W.shape)
    
    print('testing forward prop')

    X = np.arange(5*4).reshape(5,4)
    As, Zs = forward_propagate(X, Ws, bs, 'sigmoid')
    for i, A in enumerate(As):
        # print(A)
        # print(Zs[i])
        print(A.shape)
    
    Ws, bs = init_weights([5,2,3,2],1)
    As2, Zs = forward_propagate(X, Ws, bs, 'sigmoid')

    DEDWs, DEDbs = back_propagate(X, As2[-1], As, Zs, Ws, bs, final_activation_function='softmax', activation_function = 'sigmoid')

    for i, DEDW in enumerate(DEDWs):
        print(DEDW)
        print(DEDbs[i])

    print(As[-1])
    print(softmax(As[-1])[:,1])
    print(softmax(As[-1]))


    

