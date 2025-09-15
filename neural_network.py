import numpy as np
from typing import Callable, List, Tuple, Union, Any
from string import ascii_letters

class ActivationFunctions:
    functions = {
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': np.tanh,
        'relu': lambda x: np.maximum(0, x)
    }

    @classmethod
    def get(cls, name: str):
        if name not in cls.functions:
            raise ValueError(f"Unknown activation function: {name}")
        return cls.functions[name]


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
        result = np.einsum(einsum_string, self, obj2)
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

def forward_propagate(X, Ws: List[np.ndarray], bs: List[np.ndarray], activation_function = 'sigmoid'):
    layers = len(Ws)
    if layers != len(bs):
        raise ValueError('Ws and Bs need to be of the same length; the number of layers excluding the input.')
    As = []
    Zs = []
    previous_A = X
    for layer in range(layers):
        Wi = Ws[layer]
        bi = bs[layer]
        # Bi unneeded due to how numpy handles addition
        Zi = Wi.dot(previous_A) + bi
        Ai = ActivationFunctions.get(activation_function)(Zi)
        previous_A = Ai
        As.append(Ai)
        Zs.append(Zi)
    return As, Zs

    
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
        print(W)
        print(bs[i])
    
    print('testing forward prop')
    X = np.arange(5*4).reshape(5,4)
    As, Zs = forward_propagate(X, Ws, bs, 'sigmoid')
    for i, A in enumerate(As):
        print(A)
        print(Zs[i])

    

