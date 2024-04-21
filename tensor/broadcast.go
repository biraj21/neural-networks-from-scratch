package tensor

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

func (b *BroadcastTensor[T]) FlattenedGet(index int) T {
	return b.tensor.data[index%len(b.tensor.data)]
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
