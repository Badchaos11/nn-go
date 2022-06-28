package network

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNet) BackPropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	fmt.Println("Starting training")
	// Обучение модели на заданном числе эпох
	for i := 0; i < nn.Config.NumEpochs; i++ {

		// Завершаем процесс прямого распространения.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return ReLU(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Завершаем обратное расространение.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return ReLUPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Регулируем параметры.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.Config.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := SumAxes(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.Config.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.Config.LearningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := SumAxes(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.Config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

func SumAxes(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (nn *NeuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Проверяем, представляет ли значение neuralNet
	// обученную модель.
	if nn.WHidden == nil || nn.WOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.BHidden == nil || nn.BOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Определяем выход сети.
	output := new(mat.Dense)

	// Завершаем процесс прямого распространения.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applyReLU := func(_, _ int, v float64) float64 { return ReLU(v) }
	hiddenLayerActivations.Apply(applyReLU, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.WOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applyReLU, outputLayerInput)

	return output, nil
}
