package tensor

import "reflect"

// Perfoms matrix multiplication on two 2D matrices.
func MatrixMultiplication[T Scalar](t1, t2 *Tensor[T]) (result *Tensor[T]) {
	// check if both tensors are 2D matrices
	if len(t1.shape) != 2 || len(t2.shape) != 2 {
		panic("Both tensors must be 2D matrices!")
	}

	// check if the number of columns in the first matrix is equal to the number of rows in the second matrix
	if t1.shape[1] != t2.shape[0] {
		panic(ErrorMatMulConflictingDims)
	}

	resultShape := []uint{t1.shape[0], t2.shape[1]}
	result = WithShape[T](resultShape)

	for r := 0; r < int(resultShape[0]); r++ {
		for c := 0; c < int(resultShape[1]); c++ {
			sumOfProducts := T(0)
			for k := 0; k < int(t1.shape[1]); k++ {
				sumOfProducts += t1.Get(r, k) * t2.Get(k, c)
			}

			// fmt.Printf("Sum of products at (%d, %d): %v\n", r, c, sumOfProducts)
			result.Set([]int{r, c}, sumOfProducts)
		}
	}

	return result
}

// Adds two tensors.
func Add[T Scalar](t1, t2 *Tensor[T]) *Tensor[T] {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		panic(ErrorShapeMismatch)
	}

	result := WithShape[T](t1.shape)
	for i := 0; i < len(t1.data); i++ {
		result.data[i] = t1.data[i] + t2.data[i]
	}

	return result
}

// Subtracts two tensors.
func Subtract[T Scalar](t1, t2 *Tensor[T]) *Tensor[T] {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		panic(ErrorShapeMismatch)
	}

	result := WithShape[T](t1.shape)
	for i := 0; i < len(t1.data); i++ {
		result.data[i] = t1.data[i] - t2.data[i]
	}

	return result
}

// Multiplies two tensors.
func Multply[T Scalar](t1, t2 *Tensor[T]) *Tensor[T] {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		panic(ErrorShapeMismatch)
	}

	result := WithShape[T](t1.shape)
	for i := 0; i < len(t1.data); i++ {
		result.data[i] = t1.data[i] * t2.data[i]
	}

	return result
}

// Divides two tensors.
func Divide[T Scalar](t1, t2 *Tensor[T]) *Tensor[T] {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		panic(ErrorShapeMismatch)
	}

	result := WithShape[T](t1.shape)
	for i := 0; i < len(t1.data); i++ {
		result.data[i] = t1.data[i] / t2.data[i]
	}

	return result
}

// Returns the transpose of the given tensor.
func Transpose[T Scalar](t *Tensor[T]) *Tensor[T] {
	numDimensions := len(t.shape)

	// transpose of a 0D & 1D tensors is the same as itself
	if numDimensions < 2 {
		return t.Copy()
	}

	// reverse the shape of the given tensor to obtain the shape of the transposed tensor
	reversedShape := make([]uint, numDimensions)
	for i := 0; i < numDimensions; i++ {
		reversedShape[i] = t.shape[numDimensions-1-i]
	}

	// initialize the transposed Tensor
	transposedTensor := WithShape[T](reversedShape)

	// get all the indices of the original tensor
	originalIndices := getAllIndices(t.shape)

	// array that will hold the reversed location
	reversedLocation := make([]int, numDimensions)

	// traverse the original tensor and set the values to the transposed tensor
	for _, location := range originalIndices {
		// reverse the current location
		for j := 0; j < numDimensions; j++ {
			reversedLocation[j] = location[numDimensions-1-j]
		}

		// set the value at the reversed location to the transposed tensor
		transposedTensor.Set(reversedLocation, t.Get(location...))
	}

	return transposedTensor
}
