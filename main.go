package main

import (
	"fmt"

	"github.com/biraj21/nnfs-go/nn/layers"
	"github.com/biraj21/nnfs-go/tensor"
)

func main() {
	// 3 samples x 4 features
	inputs := tensor.WithValue[float64]([][]float64{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	})

	// 4 inputs x 3 neurons (basically each column is weights per neuron)
	weights1 := tensor.WithValue[float64]([][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}).Transpose()

	// 3 neurons
	biases1 := tensor.WithValue[float64]([][]float64{{2.0, 3.0, 0.5}})

	// 3 inputs x 3 neurons (3 inputs cuz prev layer has 3 neurons)
	weights2 := tensor.WithValue[float64]([][]float64{
		{0.1, -0.14, 0.5},
		{-0.5, 0.12, -0.33},
		{-0.44, 0.73, -0.13},
	}).Transpose()

	// 3 neurons
	biases2 := tensor.WithValue[float64]([][]float64{{-1, 2, -0.5}})

	l1 := layers.DenseInit(3, 4, layers.DenseInitTensors{Weights: weights1, Biases: biases1})
	l1Output := l1.Forward(inputs)

	l2 := layers.DenseInit(3, 3, layers.DenseInitTensors{Weights: weights2, Biases: biases2})
	l2Output := l2.Forward(l1Output)

	fmt.Println(l2Output)
}
