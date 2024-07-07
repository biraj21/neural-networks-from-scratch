package layers

import (
	"fmt"
	"math"
	"reflect"

	"github.com/biraj21/nnfs-go/tensor"
)

type Dense struct {
	numNeurons uint
	numInputs  uint

	// shape: numInputs x numNeurons
	weights *tensor.Tensor[float64]

	// shape: 1 x numNeurons
	biases *tensor.Tensor[float64]
}

// use this to initialize weights & biases with the given tensors
type DenseInitTensors struct {
	Weights *tensor.Tensor[float64]
	Biases  *tensor.Tensor[float64]
}

func DenseInit(numNeurons, numInputs uint, initTensors DenseInitTensors) Dense {
	layer := Dense{
		numNeurons: numNeurons,
		numInputs:  numInputs,
	}

	weightsShape := []uint{numInputs, numNeurons}
	biasesShape := []uint{1, numNeurons}

	if initTensors.Weights == nil {
		// initialize weights using Uniform Xavier initialization
		// note: i just know that this is one of the state of the art methods
		// for weight initialization. idk why. out of league rn
		x := math.Sqrt(6.0 / float64((numNeurons + numInputs)))
		layer.weights = tensor.WithRandom[float64](weightsShape, -x, x)
	} else {
		if reflect.DeepEqual(initTensors.Weights.Shape(), weightsShape) {
			layer.weights = initTensors.Weights
		} else {
			panic(fmt.Sprintf(
				"DenseInit(): expected weights of shape %v, got %v",
				weightsShape,
				initTensors.Weights.Shape(),
			))
		}
	}

	if initTensors.Biases == nil {
		// initialize with zero
		layer.biases = tensor.WithShape[float64](biasesShape)
	} else {
		if reflect.DeepEqual(initTensors.Biases.Shape(), biasesShape) {
			layer.biases = initTensors.Biases
		} else {
			panic(fmt.Sprintf(
				"DenseInit(): expected weights of shape %v, got %v",
				biasesShape,
				initTensors.Biases.Shape(),
			))
		}
	}

	return layer
}

func (d *Dense) Forward(inputs *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	output := tensor.MatrixMultiplication(inputs, d.weights).Add(d.biases)
	return output
}
