package main

import (
	"fmt"
	"log"

	"github.com/biraj21/nnfs-go/tensor"
)

func main() {
	zeroD := tensor.WithValue[int](1)
	zeroD.Set([]int{}, 10)
	fmt.Println(zeroD)

	l := tensor.WithValue[uint]([]uint{1, 2, 3, 4, 5})
	fmt.Println(l)

	// lol := tensor.WithValue[int]([][]int{
	// 	{1, 5, 6, 4},
	// 	{3, 2, 1, 3},
	// })

	// m1 := tensor.WithValue[int]([][]int{
	// 	{1, 2, 3},
	// 	{4, 5, 6},
	// })

	// m2 := tensor.WithValue[int]([][]int{
	// 	{7, 8},
	// 	{9, 10},
	// 	{11, 12},
	// })

	// fmt.Println(tensor.MatrixMultiplication(m1, m2))

	lolol := tensor.WithValue[int]([][][]int{
		{
			{1, 5, 6, 2},
			{3, 2, 1, 3},
		},
		{
			{1, 5, 6, 2},
			{3, 2, 1, 3},
		},
		{
			{1, 5, 6, 2},
			{3, 2, 1, 3},
		},
	})

	fmt.Println(lolol)

	lololol := tensor.WithValue[int]([][][][]int{
		{
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
		},
		{
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
			{
				{1, 5, 6, 2},
				{3, 2, 1, 3},
			},
		},
	})
	fmt.Println(lololol)
	return

	inputs := []float32{1.0, 2.0, 3.0, 2.5}
	weights := [][]float32{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}

	biases := []float32{2.0, 3.0, 0.5}

	// output of a single neuron/perceptron is calculated like this:
	// output := inputs[0]*weights[0] + inputs[1]*weights[1] + ... + inputs[n-1]*weights[n-1] + bias

	outputs := make([]float32, len(weights))
	for i := 0; i < len(weights); i++ {
		// make sure that the length of this neuron's weights is the same as that of the inputs
		if len(inputs) != len(weights[i]) {
			log.Fatal("The length of inputs and weights should be the same")
		}

		// calculate the output of this neuron
		outputs[i] = biases[i]
		for j := 0; j < len(inputs); j++ {
			outputs[i] += inputs[j] * weights[i][j]
		}
	}

	fmt.Println(outputs)
}
