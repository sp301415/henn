package henn

import "github.com/tuneinsight/lattigo/v4/rlwe"

// layer is a dummy interface for Convolution, Linear, and Activation layers.
type layer interface {
	isLayer()
}

// ConvLayer represents the convolution layer.
type ConvLayer struct {
	InputX int
	InputY int

	Kernel [][]float64
	Stride int
}

// isLayer implements layer interface.
func (ConvLayer) isLayer() {}

// LinearLayer represents the linear layer.
type LinearLayer struct {
	Weights [][]float64
	Bias    []float64
}

// isLayer implements layer interface.
func (LinearLayer) isLayer() {}

// ActivationLayer represents the activation layer.
type ActivationLayer struct {
	ActivationFn func(*rlwe.Ciphertext)
}

// isLayer implements layer interface.
func (ActivationLayer) isLayer() {}
