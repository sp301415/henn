package henn

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// HENeuralNet reperesents the Neural Network with Homomorphic Encryption Operations.
type HENeuralNet struct {
	Parameters ckks.Parameters
	Encoder    ckks.Encoder
	Evaluator  ckks.Evaluator
	Layers     []layer
}

// NewHENeuralNet returns the empty neural net with Encoder, Evaluator initialized.
func NewHENeuralNet(pk *PublicKeySet) *HENeuralNet {
	return &HENeuralNet{
		Parameters: pk.Parameters,
		Encoder:    ckks.NewEncoder(pk.Parameters),
		Evaluator:  ckks.NewEvaluator(pk.Parameters, pk.EvaluationKey),
	}
}

// AddLayer adds either ConvLayer, LinearLayer, and ActivationLayer to layers.
func (nn *HENeuralNet) AddLayer(l layer) {
	nn.Layers = append(nn.Layers, l)
}

// Infer executes the forward propagation, returning inferred value.
// If this network starts with ConvLayer, input should be encoded with EncryptIm2Col.
// Analogous to forward() in TenSEAL.
func (nn *HENeuralNet) Infer(input *rlwe.Ciphertext) *rlwe.Ciphertext {
	output := input.CopyNew()

	for _, l := range nn.Layers {
		switch l := l.(type) {
		case ConvLayer:
			nn.conv(l, output)
		case LinearLayer:
		case ActivationLayer:
		default:
			panic("unknown layer")
		}
	}

	return output
}

// conv executes ConvLayer in-place.
// TODO
func (nn *HENeuralNet) conv(l ConvLayer, ct *rlwe.Ciphertext) {
	kx := len(l.Kernel)
	ky := len(l.Kernel[0])

	repeat := ((l.InputX - kx + l.Stride) / l.Stride) * ((l.InputY - ky + l.Stride) / l.Stride)
	flattenedKernel := make([]float64, 0, repeat*kx*ky)
	for i := 0; i < kx; i++ {
		for j := 0; j < ky; j++ {
			for n := 0; n < repeat; n++ {
				flattenedKernel = append(flattenedKernel, l.Kernel[kx][ky])
			}
		}
	}

	kernelPlaintext := nn.Encoder.EncodeNew(flattenedKernel, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())

	nn.Evaluator.MulRelin(ct, kernelPlaintext, ct)

}
