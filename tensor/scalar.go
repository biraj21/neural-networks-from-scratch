package tensor

import (
	"fmt"
	"reflect"
)

// _NumericScalar is a type that is a numeric scalar. For example, int, float64, etc.
type _NumericScalar interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 | uintptr |
		float32 | float64 |
		complex64 | complex128
}

// Scalar is a type that is only a single value, not a collection of values. For example, int, float64, etc.
//
// Note that it doesn't include booleans becuase they are not numeric, and thus numeric operations can't be performed on them.
type Scalar interface {
	_NumericScalar
}

// Checks if the provided value is a Scalar or not. Panics if it's a Scalar but not of the expected type.
func IsScalar[T interface{}](value interface{}) bool {
	val := reflect.ValueOf(value)
	switch val.Kind() {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.Complex64, reflect.Complex128:

		switch val.Interface().(type) {
		case T:
			// eat 5-star, do nothing
		default:
			var validScalar T
			panic(fmt.Sprintf("Data type mismatch! Expected a Scalar of type %T, found %T", validScalar, val.Interface()))
		}
		return true
	default:
		return false
	}
}
