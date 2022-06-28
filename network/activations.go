package network

import (
	"math"
)

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func Linear(x float64) float64 {
	return x
}

func LinearPrime(x float64) float64 {
	return 1
}
