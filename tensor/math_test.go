package tensor

import (
	"reflect"
	"testing"
)

var t1 = WithValue[int]([]int{1, 2, 3})
var t2 = WithValue[int]([][]int{
	{1},
	{2},
	{3},
})

func TestAdd(t *testing.T) {
	expected := WithValue[int]([][]int{
		{2, 3, 4},
		{3, 4, 5},
		{4, 5, 6},
	})

	result := Add(t1, t2)
	if !reflect.DeepEqual(expected, result) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestSubtract(t *testing.T) {
	expected := WithValue[int]([][]int{
		{0, 1, 2},
		{-1, 0, 1},
		{-2, -1, 0},
	})

	result := Subtract(t1, t2)
	if !reflect.DeepEqual(expected, result) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestMultiply(t *testing.T) {
	expected := WithValue[int]([][]int{
		{1, 2, 3},
		{2, 4, 6},
		{3, 6, 9},
	})

	result := Multiply(t1, t2)
	if !reflect.DeepEqual(expected, result) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestDivide(t *testing.T) {
	expected := WithValue[int]([][]int{
		{1, 2, 3},
		{0, 1, 1},
		{0, 0, 1},
	})

	result := Divide(t1, t2)
	if !reflect.DeepEqual(expected, result) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}
