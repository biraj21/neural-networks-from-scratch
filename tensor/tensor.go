package tensor

import (
	"fmt"
	"reflect"
)

// Tensor is a struct that represents a multi-dimensional array.
type Tensor[T _NumericScalar] struct {
	data     []T
	dataType reflect.Type
	shape    []uint
	strides  []uint
}

// Returns the value of the tensor.
func (t *Tensor[T]) Value() interface{} {
	return t.data
}

// Returns the shape of the tensor.
func (t *Tensor[T]) Shape() []uint {
	return t.shape
}

// Returns the number of dimensions of the tensor.
func (t *Tensor[T]) NDims() int {
	return len(t.shape)
}

// Returns the data type of the tensor.
func (t *Tensor[T]) DataType() reflect.Type {
	return t.dataType
}

func (t *Tensor[T]) Copy() *Tensor[T] {
	dataCopy := make([]T, len(t.data))
	copy(dataCopy, t.data)

	return &Tensor[T]{
		data:     dataCopy,
		dataType: t.dataType,
		shape:    t.shape,
		strides:  t.strides,
	}
}

func (t *Tensor[T]) Reshape(newDims ...uint) *Tensor[T] {
	if len(newDims) == 0 {
		panic("Cannot reshape to an empty shape!")
	}

	// make sure that reshaping is possible
	if countElementsFromShape(newDims) != countElementsFromShape(t.shape) {
		panic(fmt.Sprintf("Incompatible reshaping: %v -> %v", t.shape, newDims))
	}

	// create a copy of this tensor
	tCopy := t.Copy()

	// update the shape and strides
	tCopy.shape = newDims
	tCopy.strides = calculateStrides(newDims)

	return tCopy
}

func (t *Tensor[T]) indicesToDataIndex(indices ...int) int {
	dataIndex := 0
	for i, index := range indices {
		dataIndex += index * int(t.strides[i])
	}

	return dataIndex
}

func (t *Tensor[T]) Get(indices ...int) T {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("Invalid number of indices %d for tensor of shape %v", len(indices), t.shape))
	}

	index := t.indicesToDataIndex(indices...)
	return t.data[index]
}

func (t *Tensor[T]) Set(indices []int, value T) {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("Invalid number of indices %d for tensor of shape %v", len(indices), t.shape))
	}

	index := t.indicesToDataIndex(indices...)
	t.data[index] = value
}

// Returns a string representation of the tensor.
func (t *Tensor[T]) String() string {
	s := fmt.Sprintf("{\n  shape: %v\n  dataType: %v\n  value: %v", t.shape, t.dataType, t.data)
	// if t.NDims() > 1 {
	// 	s += "\n" + prettifyTensorValue(t.data, 2)
	// } else {
	// 	s += " " + prettifyTensorValue(t.data)
	// }

	s += "\n}"
	return s
}

// Adds two tensors.
func (t *Tensor[T]) Add(t2 *Tensor[T]) *Tensor[T] {
	return Add(t, t2)
}

// Subtracts two tensors.
func (t *Tensor[T]) Subtract(t2 *Tensor[T]) *Tensor[T] {
	return Subtract(t, t2)
}

// Multiplies two tensors.
func (t *Tensor[T]) Multply(t2 *Tensor[T]) *Tensor[T] {
	return Multply(t, t2)
}

// Divides two tensors.
func (t *Tensor[T]) Divide(t2 *Tensor[T]) *Tensor[T] {
	return Divide(t, t2)
}

// Creates a new tensor with the given shape and initial value.
func WithShape[T Scalar](shape []uint, initialValue ...T) *Tensor[T] {
	if len(initialValue) > 1 {
		panic("Only one initial value is allowed!")
	}

	for i, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("Invalid shape: dimension %d cannot be %d", i, dim))
		}
	}

	data := make([]T, countElementsFromShape(shape))
	if len(initialValue) > 0 {
		for i := 0; i < int(shape[0]); i++ {
			data[i] = initialValue[0]
		}
	}

	// dummy variable to get the data type at runtime
	var dataType T

	return &Tensor[T]{
		data:     data,
		dataType: reflect.TypeOf(dataType),
		shape:    shape,
		strides:  calculateStrides(shape),
	}
}

// Create a new tensor from the given value.
func WithValue[T Scalar](data interface{}) *Tensor[T] {
	// validate that the tensor is homogenous, i.e., all elements are of the same type
	ensureHomogeneous[T](data)

	// validate that the tensor is homologous, i.e. each list along a dimension is of the same size
	ensureHomologous(data)

	// determine the shape of the tensor
	shape := detectShape(data)

	numberOfElements := countElementsFromShape(shape)

	// it's length would be same as the number of elements in the tensor
	tensorIndices := getAllIndices(shape)

	tensorData := make([]T, numberOfElements)
	for i := uint(0); i < numberOfElements; i++ {
		tensorData[i] = valueAt(data, tensorIndices[i]...).Interface().(T)
	}

	// just dummy variable to get the data type at runtime
	var dataType T

	// create the tensor & return
	return &Tensor[T]{
		data:     tensorData,
		dataType: reflect.TypeOf(dataType),
		shape:    shape,
		strides:  calculateStrides(shape),
	}
}
