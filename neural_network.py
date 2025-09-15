import numpy as np
from typing import Callable, List, Tuple, Union, Any
from string import ascii_letters


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
    mid_string_reverse = mid_string[::-1]
    letters = letters[tensor_dim_denominator1:]
    right_string = letters[0:tensor_dim_denominator2]
    return left_string + mid_string + ',' + mid_string_reverse +  right_string + '->' + left_string + right_string

    
if __name__ == "__main__":
    a = np.arange(2*4*5).reshape(2,4,5)
    print(a)
    b = np.arange(4*5*9).reshape(5,4,3,3)
    c = np.einsum(_get_einsum_string_for_mul_tensor_jacobian([1,0],[1,1],[1,1],[1,1]),a,b)
    print(c)
    print(c.shape)
    print(ascii_letters)
    print (a.reshape(1,-1))
    print(_get_einsum_string_for_mul_tensor_jacobian([1,0],[1,1],[1,1],[1,1]))
    Ta = TensorJacobian(a,[1,0],[1,1])
    Tb = TensorJacobian(b,[1,1],[1,1])
    print(Ta*Tb)

    

