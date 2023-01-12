package henn

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// Layer is a dummy interface for Convolution, Linear, and Activation layers.
type Layer interface {
	isLayer()
}

// ConvLayer represents the convolution layer.
type ConvLayer struct {
	InputX int
	InputY int

	Kernel [][][]float64
	Bias   []float64
	Stride int
}

// isLayer implements Layer interface.
func (ConvLayer) isLayer() {}

// LinearLayer represents the linear layer.
type LinearLayer struct {
	Weights [][]float64
	Bias    []float64
}

// isLayer implements Layer interface.
func (LinearLayer) isLayer() {}

// ActivationLayer represents the activation layer.
type ActivationLayer struct {
	ActivationFn func(*HENeuralNet, *rlwe.Ciphertext)
}

// isLayer implements Layer interface.
func (ActivationLayer) isLayer() {}

// isEncodedLayer implements EncodedLayer interface.
func (ActivationLayer) isEncodedLayer() {}

// EncodedLayer represents the encoded layers that can be directly used in HENeuralNets.
type EncodedLayer interface {
	isEncodedLayer()
}

// EncodedConvLayer represents the encoded convolution layer.
type EncodedConvLayer struct {
	Im2ColX int // Same as kernel size
	Im2ColY int // Same as repeats
	mask    *rlwe.Plaintext

	Kernel []*rlwe.Plaintext
	Bias   []*rlwe.Plaintext
	Stride int
}

// isEncodedLayer implements EncodedLayer interface.
func (EncodedConvLayer) isEncodedLayer() {}

// EncodedLinearLayer represents the encoded linear layer.
type EncodedLinearLayer struct {
	Weights ckks.LinearTransform
	Bias    *rlwe.Plaintext
}

// isEncodedLayer implements EncodedLayer interface.
func (EncodedLinearLayer) isEncodedLayer() {}
