package tensor

import (
	"reflect"
	"testing"
)

func TestBroadcast(t *testing.T) {
	t1 := WithValue[int]([]int{1, 2, 3})
	t2 := WithValue[int]([][]int{
		{1},
		{2},
		{3},
	})

	expectedLen := 2
	broadcasts := Broadcast(t1, t2)
	if expectedLen != len(broadcasts) {
		t.Fatalf("len(broadcasts): expected %v, found %v", expectedLen, len(broadcasts))
	}

	expectedBt1 := WithValue[int]([][]int{
		{1, 2, 3},
		{1, 2, 3},
		{1, 2, 3},
	})
	bt1 := broadcasts[0].ToTensor()
	if !reflect.DeepEqual(expectedBt1, bt1) {
		t.Fatalf("broadcasts[0].ToTensor(): expected %v, found %v", expectedBt1, bt1)
	}

	expectedBt2 := WithValue[int]([][]int{
		{1, 1, 1},
		{2, 2, 2},
		{3, 3, 3},
	})
	bt2 := broadcasts[1].ToTensor()
	if !reflect.DeepEqual(expectedBt2, bt2) {
		t.Fatalf("broadcasts[1].ToTensor(): expected %v, found %v", expectedBt2, bt2)
	}
}
