package network

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	Config  NeuralNetConfig
	WHidden *mat.Dense
	BHidden *mat.Dense
	WOut    *mat.Dense
	BOut    *mat.Dense
}

type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
	LearningRate  float64
}

func NewNetwork(config NeuralNetConfig) *NeuralNet {
	fmt.Println("Creating new NN")
	return &NeuralNet{Config: config}
}
