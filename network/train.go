package network

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNet) Train(x, y *mat.Dense) error {
	fmt.Println("Starting train network...")

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	fmt.Println(randGen)

	wHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, nil)
	wOut := mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, nil)
	bOut := mat.NewDense(1, nn.Config.OutputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}
	output := new(mat.Dense)

	if err := nn.BackPropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Определяем обученную сеть.
	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut

	return nil
}
