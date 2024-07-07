package tensor

import (
	"fmt"
	"math/rand/v2"
	"reflect"
	"strings"
)

// Recursively initializes the tensor with the given shape and initial value.
func initTensorValue[T Scalar](shape []uint, initialValue ...T) interface{} {
	if len(initialValue) > 1 {
		panic("There cannot be more than one initial value for a tensor!")
	}

	if len(shape) == 0 {
		if len(initialValue) == 0 {
			var emptyValue T
			return emptyValue
		}

		return initialValue[0]
	}

	if len(shape) == 1 {
		slice := make([]T, shape[0])

		if len(initialValue) > 0 {
			for i := range slice {
				slice[i] = initialValue[0]
			}
		}

		return slice
	}

	slice := make([]interface{}, shape[0])
	for i := range slice {
		slice[i] = initTensorValue(shape[1:], initialValue...)
	}

	return slice
}

// Recursively validates that the values in the tensor are of Scalars and of the same type.
func ensureHomogeneous[T Scalar](value interface{}) {
	val := reflect.ValueOf(value)
	kind := val.Kind()

	// if it's a Scalar, then just make sure that it's of type T
	if IsScalar[T](value) {
		return
	}

	// it should be a multi-dimensional array or slice
	if kind != reflect.Array && kind != reflect.Slice {
		panic(ErrorNonArraySlice)
	}

	// it should not be empty
	if val.Len() == 0 {
		panic(ErrorEmptyArraySlice)
	}

	// validate each element's type
	for i := 0; i < val.Len(); i++ {
		elem := val.Index(i)
		if elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice {
			ensureHomogeneous[T](elem.Interface())
			continue
		}

		if IsScalar[T](elem.Interface()) {
			continue
		}

		var validScalar T
		panic(fmt.Sprintf("Unexpected type %T in tensorand. Expected a Scalar of type %T", elem.Interface(), validScalar))
	}
}

// Checks if the provided tensor value is homologous. Panics with an error message if it's not.
// It first detects and ensure that the tensor matches the shape. Basically, it's a wrapper over detectShape() and ensureShape() functions.
func ensureHomologous(value interface{}) {
	shape := detectShape(value)

	// if it's just a scalar or single-dimensional, then it's obviously homologous
	if len(shape) < 2 {
		return
	}

	ensureShape(value, shape, 0)
}

// Tries to detect the shape of the tensor value.
func detectShape(value interface{}) (shape []uint) {
	val := reflect.ValueOf(value)
	shape = []uint{}

	for {
		// validate that it's an array or slice
		valueKind := val.Kind()
		if valueKind != reflect.Array && valueKind != reflect.Slice {
			break
		}

		// validate that it's not empty
		if val.Len() == 0 {
			panic(ErrorEmptyArraySlice)
		}

		// append the size of the current dimension
		shape = append(shape, uint(val.Len()))

		// go deeper to get next dimension's size
		val = val.Index(0)
	}

	return shape
}

func ensureShape(value interface{}, shape []uint, currentDim int) {
	if len(shape) == 0 {
		panic("Invalid shape")
	}

	val := reflect.ValueOf(value)
	if val.Kind() != reflect.Array && val.Kind() != reflect.Slice {
		panic(ErrorNonArraySlice)
	}

	if uint(val.Len()) != shape[currentDim] {
		panic(fmt.Sprintf("%s Detected shape was %v, but there's a mismatch at dimension %d with size %d. Shouldn't it be of size %d?",
			ErrorNonHomologous,
			shape,
			currentDim,
			val.Len(),
			shape[currentDim],
		))
	}

	for i := 0; i < val.Len(); i++ {
		elem := val.Index(i)

		if elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice {
			ensureShape(elem.Interface(), shape, currentDim+1)
			continue
		} else if currentDim < len(shape)-1 {
			panic(fmt.Sprintf("%s Detected shape was %v, but found an unexpected %d at dimension %d. Expected a slice or array.",
				ErrorNonHomologous,
				shape,
				elem.Type(),
				currentDim,
			))
		}
	}
}

// Returns a pretty string representation of the tensor value.
func prettifyTensorValue(value interface{}, indentation ...int) string {
	if IsScalar[any](value) {
		return fmt.Sprintf("%v", value)
	}

	val := reflect.ValueOf(value)
	if val.Kind() != reflect.Array && val.Kind() != reflect.Slice {
		panic(ErrorNonArraySlice)
	}

	if len(indentation) == 0 {
		indentation = []int{0}
	}

	indentationLevel := indentation[0]
	indentationStr := strings.Repeat(" ", indentationLevel)
	if val.Len() == 0 {
		return indentationStr + "[]"
	}

	isDeepest := true
	s := indentationStr + "["
	for i := 0; i < val.Len(); i++ {
		elem := val.Index(i)

		if elem.Kind() == reflect.Interface {
			elem = elem.Elem()
		}

		if elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice {
			s += "\n" + prettifyTensorValue(elem.Interface(), indentationLevel+2)
			isDeepest = false
		} else {
			s += fmt.Sprintf("%v", elem.Interface())
		}

		// add comma if it's not the last element
		if i < val.Len()-1 {
			s += " "
		}

		// add new line if it's an array or slice
		if (elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice) && i == val.Len()-1 {
			s += "\n"
		}
	}

	/// only add indentation before closing bracket if it's not the deepest level
	if !isDeepest {
		s += indentationStr
	}

	s += "]"
	return s
}

func calculateStrides(shape []uint) []uint {
	// if it's a zero-d array, then there's no strides
	if len(shape) == 0 {
		return []uint{}
	}

	// initialize strides. it's the same length as the shape
	strides := make([]uint, len(shape))

	// last stride is always 1
	strides[len(shape)-1] = 1

	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = shape[i+1] * strides[i+1]
	}

	return strides
}

func valueAt(data interface{}, indices ...int) reflect.Value {
	val := reflect.ValueOf(data)

	// handle Scalars
	if len(indices) == 1 && IsScalar[any](data) {
		return val
	}

	if val.Kind() != reflect.Slice && val.Kind() != reflect.Array {
		panic(ErrorNonArraySlice)
	}

	for i, index := range indices {
		if val.Kind() == reflect.Interface {
			val = val.Elem()
		}

		if val.Kind() == reflect.Slice || val.Kind() == reflect.Array {
			val = val.Index(index)
		} else {
			panic(fmt.Sprintf("Invalid tensor value %v at %v", data, indices[:i+1]))
		}
	}

	return val
}

// Returns all the traversable indices for a tenson of the given shape.
//
// For example, if the shape is [2, 3], then the returned indices will be:
// [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
func getAllIndices(shape []uint) [][]int {
	numDimensions := len(shape)
	numElements := countElementsFromShape(shape)

	tensorIndices := make([][]int, numElements)

	if numDimensions == 0 {
		tensorIndices[0] = []int{0}
		return tensorIndices
	}

	indices := make([]int, numDimensions)

	// initialize first set of indices to 0 (eg. [0, 0, ..., 0]) it will be added to tensorIndices in the loop
	for i := range indices {
		indices[i] = 0
	}

	// populate the tensor indices in a pattern like
	// [0, 0, 0], [0, 0, 1], [0, 0, 2], ..., [0, 1, 0], [0, 1, 1], [0, 1, 2], ..., [1, 0, 0], [1, 0, 1], [1, 0, 2], ...
	for i := 0; i < len(tensorIndices); i++ {
		tensorIndices[i] = make([]int, numDimensions)
		copy(tensorIndices[i], indices)

		// increment the indices
		for dim := numDimensions - 1; dim >= 0; dim-- {
			indices[dim]++

			// if the index at current dimension is still less than its size, we will keep on
			// incrementing that dimension's index only before moving to the next one
			if indices[dim] < int(shape[dim]) {
				break
			}

			// reset the current dimension's index to 0 before moving to the next dimension
			indices[dim] = 0
		}
	}

	return tensorIndices
}

func countElementsFromShape(shape []uint) uint {
	count := uint(1)
	for _, dimSize := range shape {
		count *= dimSize
	}

	return count
}

func areShapesBroadcastable(shapes ...[]uint) bool {
	if len(shapes) == 0 {
		panic("areShapesBroadcastable: At least one shape is required")
	}

	maxDimensions := 0
	for _, shape := range shapes {
		if len(shape) > maxDimensions {
			maxDimensions = len(shape)
		}
	}

	firstShape := make([]uint, maxDimensions)
	copyWithPadding(firstShape, shapes[0], 1)

	currentShape := make([]uint, maxDimensions)
	for i := 1; i < len(shapes); i++ {
		copyWithPadding(currentShape, shapes[i], 1)

		for j := 0; j < maxDimensions; j++ {
			// general broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
			// two dimensions are compatible if 1. they are equal, or 2. one of them is 1
			f := firstShape[j]
			c := currentShape[j]
			if f != c && f != 1 && c != 1 {
				return false
			}
		}
	}

	return true
}

func copyWithPadding[T Scalar](dest []T, src []T, padWith T) {
	if len(dest) < len(src) {
		panic("Length of destination array cannot be lesser than that of source array.")
	}

	// add padding
	for i := 0; i < len(dest)-len(src); i++ {
		dest[i] = padWith
	}

	// copy the remaining values
	copy(dest[len(dest)-len(src):], src)
}

func randomBetween[T Scalar](minValue, maxValue T) T {
	switch any(minValue).(type) {
	case int, int8, int16, int32, int64:
		return T(rand.Int64N(int64(maxValue-minValue)) + int64(minValue))
	case uint, uint8, uint16, uint32, uint64, uintptr:
		return T(rand.Uint64N(uint64(maxValue-minValue)) + uint64(minValue))
	case float32:
		return T(rand.Float32()*float32(maxValue-minValue) + float32(minValue))
	case float64:
		return T(rand.Float64()*float64(maxValue-minValue) + float64(minValue))
	default:
		panic("Unsupported type for random number generation")
	}
}
