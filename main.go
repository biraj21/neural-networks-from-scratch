package main

import (
	"fmt"

	"github.com/biraj21/nnfs-go/tensor"
)

func main() {
	inputs := tensor.WithValue[float64]([][]float64{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	})

	weights := tensor.WithValue[float64]([][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	})

	biases := tensor.WithValue[float64]([]float64{2.0, 3.0, 0.5})

	result := tensor.MatrixMultiplication(inputs, weights.Transpose()).Add(biases)

	fmt.Println(result)
}
