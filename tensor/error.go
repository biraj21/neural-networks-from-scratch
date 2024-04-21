package tensor

const (
	// Error message for an empty array or slice.
	ErrorEmptyArraySlice = "Found an empty array or slice!"

	// Error message for a value that's neither an array nor slice.
	ErrorNonArraySlice = "Value is neither an array nor a slice!"

	// Error message for a non-homologous tensor value.
	ErrorNonHomologous = "Tensor is not homologous!"

	// Error message for a non-2D matrix.
	ErrorMatMulConflictingDims = "The number of columns in the first matrix must be equal to the number of rows in the second matrix!"

	// Error message for a shape mismatch.
	ErrorShapeMismatch = "Shapes of the tensors do not match!"

	// Error message for incompatible reshaping.
	ErrorIncompatibleReshape = "Incompatible reshaping!"

	// Error message for tensors incompatible for broadcast
	ErrorCannotBroadcast = "Tensors could not be broadcast together!"
)
