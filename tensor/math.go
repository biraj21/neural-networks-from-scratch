package tensor

// Perfoms matrix multiplication on two 2D matrices.
func MatrixMultiplication[T NumericScalar](t1, t2 *Tensor[T]) (result *Tensor[T]) {
	// check if both tensors are 2D matrices
	if len(t1.shape) != 2 || len(t2.shape) != 2 {
		panic("Both tensors must be 2D matrices!")
	}

	// check if the number of columns in the first matrix is equal to the number of rows in the second matrix
	if t1.shape[1] != t2.shape[0] {
		panic(ErrorMatMulConflictingDims)
	}

	resultShape := []uint{t1.shape[0], t2.shape[1]}
	result = WithShape(resultShape, T(0))

	for r := 0; r < int(resultShape[0]); r++ {
		for c := 0; c < int(resultShape[1]); c++ {
		}
	}

	return result
}
