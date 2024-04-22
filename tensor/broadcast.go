package tensor

import "fmt"

type BroadcastTensor[T Scalar] struct {
	// new shape after broadcast
	shape []uint

	// the tensor which this broadcast represents
	tensor *Tensor[T]
}

// Returns the shape of the BroadcastTensor.
func (b *BroadcastTensor[T]) Shape() []uint {
	return b.shape
}

func (b *BroadcastTensor[T]) Get(indices ...int) T {
	if len(indices) != len(b.shape) {
		panic(fmt.Sprintf("Invalid number of indices %d for broadcast of shape %v", len(indices), b.shape))
	}

	tensorIndices := make([]int, len(b.tensor.shape))

	// skip extra indices. for eg, broadcast has 5 dims & actual tensor has just 3, then skip the first 2 indices
	copy(tensorIndices, indices[len(indices)-len(b.tensor.shape):])

	// make remaining indices compatible with the tensor's shape
	for i, dimSize := range b.tensor.shape {
		if int(dimSize) <= tensorIndices[i] {
			tensorIndices[i] = int(dimSize) - 1
		}
	}

	return b.tensor.Get(tensorIndices...)
}

func (b *BroadcastTensor[T]) dataIndexToIndices(dataIndex int) []int {
	strides := calculateStrides(b.shape)

	indices := make([]int, len(strides))
	for i := len(strides) - 1; i >= 0; i-- {
		// Calculate the index for the current dimension
		indices[i] = dataIndex % int(b.shape[i])
		// Update the remaining dataIndex for the next dimension
		dataIndex /= int(b.shape[i])
	}

	return indices
}

func (b *BroadcastTensor[T]) FlattenedGet(index int) T {
	indices := b.dataIndexToIndices(index)
	return b.Get(indices...)
}

// Converts the BroadcastTensor to Tensor.
func (b *BroadcastTensor[T]) ToTensor() *Tensor[T] {
	t := WithShape[T](b.shape)

	for _, indices := range getAllIndices(b.shape) {
		t.Set(indices, b.Get(indices...))
	}

	return t
}

func CanBroadcast[T Scalar](tensors ...*Tensor[T]) bool {
	if len(tensors) == 0 {
		panic("At least one tensor is required!")
	}

	if len(tensors) == 1 {
		return true
	}

	shapes := make([][]uint, len(tensors))

	for i, t := range tensors {
		shapes[i] = t.shape
	}

	return areShapesBroadcastable(shapes...)
}

func Broadcast[T Scalar](tensors ...*Tensor[T]) []*BroadcastTensor[T] {
	if len(tensors) == 0 {
		return []*BroadcastTensor[T]{}
	}

	if !CanBroadcast(tensors...) {
		panic(ErrorCannotBroadcast)
	}

	maxDimensions := 0
	for _, t := range tensors {
		if len(t.shape) > maxDimensions {
			maxDimensions = len(t.shape)
		}
	}

	broadcastShape := make([]uint, maxDimensions)
	for i := 0; i < maxDimensions; i++ {
		broadcastShape[i] = 0
	}

	for _, t := range tensors {
		for j, size := range t.shape {
			i := maxDimensions - len(t.shape) + j
			if size > broadcastShape[i] {
				broadcastShape[i] = size
			}
		}
	}

	broadcast := make([]*BroadcastTensor[T], len(tensors))
	for i, t := range tensors {
		broadcast[i] = &BroadcastTensor[T]{
			shape:  broadcastShape,
			tensor: t,
		}
	}

	return broadcast
}
