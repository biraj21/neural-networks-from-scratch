package tensor

import (
	"fmt"
	"reflect"
)

// Tensor is a struct that represents a multi-dimensional array.
type Tensor[T Scalar] struct {
	value    interface{}
	shape    []uint
	dataType reflect.Type
}

func (t *Tensor[T]) String() string {
	s := fmt.Sprintf("{\n  shape: %v\n  data type: %v\n  value:", t.shape, t.dataType)
	if t.NDims() > 1 {
		s += "\n" + prettifyTensorValue(t.value, 2)
	} else {
		s += " " + prettifyTensorValue(t.value)
	}

	s += "\n}\n"
	return s
}

// Returns the value of the tensor.
func (t *Tensor[T]) Value() interface{} {
	return t.value
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

// Returns the value of the tensor at the given indices.
func (t *Tensor[T]) Index(indices ...int) interface{} {
	if len(indices) > len(t.shape) {
		panic(fmt.Sprintf("Invalid number of indices %d for tensor of shape %v", len(indices), t.shape))
	}

	// get the value
	val := reflect.ValueOf(t.value)
	for _, index := range indices {
		val = val.Index(index)
	}

	// return the value as an type T
	return val.Interface()
}

// Sets a Scalar value at the given location (indices) in the tensor.
func (t *Tensor[T]) Set(indices []int, value T) {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("Invalid number of indices %d for tensor of shape %v", len(indices), t.shape))
	}

	// handle zero-dimensional tensor
	if len(t.shape) == 0 {
		t.value = value
		return
	}

	// get the value
	val := reflect.ValueOf(t.value)
	for _, index := range indices {
		val = val.Index(index)
	}

	// set the value
	val.Set(reflect.ValueOf(value))
}

// Creates a new tensor with the given shape and initial value.
func WithShape[T Scalar](shape []uint, initialValue T) *Tensor[T] {

	if len(shape) == 0 {
		return &Tensor[T]{value: initialValue, shape: shape, dataType: reflect.TypeOf(initialValue)}
	}

	for i, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("Invalid shape: dimension %d cannot be %d", i, dim))
		}

	}

	value := initTensorValue(shape, initialValue)

	return &Tensor[T]{value: value, shape: shape, dataType: reflect.TypeOf(initialValue)}
}

// Create a new tensor from the given value.
func WithValue[T Scalar](value interface{}) *Tensor[T] {
	// validate that the tensor is homogenous, i.e., all elements are of the same type
	ensureHomogeneous[T](value)

	// validate that the tensor is homologous, i.e. each list along a dimension is of the same size
	ensureHomologous(value)

	// determine the shape of the tensor
	shape := detectShape(value)

	// create the tensor
	t := Tensor[T]{value: value, shape: shape}

	// determine the data type of the tensor
	numDims := len(t.shape)
	firstElemIndex := []int{}
	for i := 0; i < numDims; i++ {
		firstElemIndex = append(firstElemIndex, 0)
	}

	firstElem := t.Index(firstElemIndex...)
	t.dataType = reflect.TypeOf(firstElem)

	return &t
}
