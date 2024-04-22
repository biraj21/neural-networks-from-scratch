package tensor

import (
	"reflect"
	"testing"
)

func TestCopyWithPadding(t *testing.T) {
	padWith := -2

	arr := []int{1, 2, 3}
	dest := make([]int, 5)

	copyWithPadding(dest, arr, padWith)

	expected := []int{padWith, padWith, 1, 2, 3}
	if !reflect.DeepEqual(expected, dest) {
		t.Fatalf("copyWithPadding(): expected %v, got %v", expected, dest)
	}
}

func TestAreShapesBroadcastableYes(t *testing.T) {
	areBroadcastable := areShapesBroadcastable(
		[]uint{5, 1},
		[]uint{1, 6},
		[]uint{6},
		[]uint{},
	)

	expected := true
	if expected != areBroadcastable {
		t.Fatalf("areBroadcastable(): expected %v, got %v", expected, areBroadcastable)
	}
}

func TestAreShapesBroadcastableNo(t *testing.T) {
	areBroadcastable := areShapesBroadcastable(
		[]uint{4, 3},
		[]uint{4},
	)

	expected := false
	if expected != areBroadcastable {
		t.Fatalf("areBroadcastable(): expected %v, got %v", expected, areBroadcastable)
	}
}
