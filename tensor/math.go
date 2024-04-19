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
