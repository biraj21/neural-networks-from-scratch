package main

import (
	"fmt"

	"github.com/biraj21/nnfs-go/tensor"
)

func main() {
	inputs := tensor.WithValue[float64]([]float64{1.0, 2.0, 3.0, 2.5})

	weights := tensor.WithValue[float64]([][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	})

	biases := tensor.WithValue[float64]([][]float64{
		{2.0},
		{3.0},
		{0.5},
	})

	result := tensor.MatrixMultiplication(weights, inputs.Reshape(4, 1)).Add(biases)

	fmt.Println(result)
}
