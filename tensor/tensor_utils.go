package tensor

import (
	"fmt"
	"reflect"
	"strings"
)

// Recursively initializes the tensor with the given shape and initial value.
func initTensorValue[T Scalar](shape []uint, initialValue T) interface{} {
	if len(shape) == 0 {
		return initialValue
	}

	slice := make([]interface{}, shape[0])
	for i := range slice {
		slice[i] = initTensorValue(shape[1:], initialValue)
	}
	return slice
}

// Recursively validates that the values in the tensor are of Scalars and of the same type.
func ensureHomogeneous[T Scalar](value interface{}) {
	val := reflect.ValueOf(value)
	kind := val.Kind()

	// if it's a Scalar, then we're good
	if IsScalar(value) {
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

		switch elem.Interface().(type) {
		case T:
			// eat 5-star, do nothing
		default:
			var validScalar T
			panic(fmt.Sprintf("Data type mismatch in tensor. Expected a Scalar of type %T, found %T", validScalar, elem.Interface()))
		}
	}
}

// Checks if the provided tensor value is homologous. Panics with an error message if it's not.
// It's a wrapper around _ensureShape() function, which is actually responsible for the validation.
func ensureHomologous(value interface{}) {
	shape := detectShape(value)

	// if it's just a scalar or single-dimensional, then it's obviously homologous
	if len(shape) < 2 {
		return
	}

	_ensureShape(value, shape, 0)
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

func _ensureShape(value interface{}, shape []uint, currentDim int) {
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
			_ensureShape(elem.Interface(), shape, currentDim+1)
			continue
		} else if currentDim < len(shape)-1 {
			panic(fmt.Sprintf("%s Detected shape was %v, but found an unexpected scalar at dimension %d",
				ErrorNonHomologous,
				shape,
				currentDim,
			))
		}
	}
}

// Returns a pretty string representation of the tensor value.
func prettifyTensorValue(value interface{}, indentation ...int) string {
	if IsScalar(value) {
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
		if elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice {
			s += "\n" + prettifyTensorValue(elem.Interface(), indentationLevel+2)
			isDeepest = false
		} else {
			s += fmt.Sprintf("%v", elem.Interface())
		}

		// add comma if it's not the last element
		if i < val.Len()-1 {
			s += ", "
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
