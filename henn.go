package henn

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// HENeuralNet represents the Neural Network with Homomorphic Encryption Operations.
type HENeuralNet struct {
	Parameters   ckks.Parameters
	PublicKeySet *PublicKeySet
	Encoder      ckks.Encoder
	Evaluator    ckks.Evaluator
	Layers       []EncodedLayer
}

// NewHENeuralNet returns the empty HENeuralNet with Encoder, Evaluator initialized.
func NewHENeuralNet(pk *PublicKeySet) *HENeuralNet {
	return &HENeuralNet{
		Parameters: pk.Parameters,
		Encoder:    ckks.NewEncoder(pk.Parameters),
		Evaluator:  ckks.NewEvaluator(pk.Parameters, pk.EvaluationKey),
	}
}

// AddLayers adds layers to this HENeuralNet.
func (nn *HENeuralNet) AddLayers(layers ...Layer) {
	for _, l := range layers {
		switch l := l.(type) {
		case ConvLayer:
			nn.Layers = append(nn.Layers, nn.EncodeConvLayer(l))
		case LinearLayer:
			nn.Layers = append(nn.Layers, nn.EncodeLinearLayer(l))
		case ActivationLayer:
			nn.Layers = append(nn.Layers, l)
		}
	}
}

// Infer executes the forward propagation, returning inferred value.
// If this network starts with ConvLayer, input should be encoded with EncryptIm2Col.
// Analogous to forward() in TenSeal.
func (nn *HENeuralNet) Infer(input *rlwe.Ciphertext) *rlwe.Ciphertext {
	output := input.CopyNew()

	for _, l := range nn.Layers {
		switch l := l.(type) {
		case EncodedConvLayer:
			nn.conv(l, output)
		case EncodedLinearLayer:
			nn.linear(l, output)
		case ActivationLayer:
			nn.activate(l, output)
		}
	}

	return output
}

// isMatrix returns if given 2D slice can be interpreted as a matrix
// i.e. All rows are in same length.
func isMatrix[T any](s [][]T) bool {
	N := len(s[0])
	for _, row := range s {
		if len(row) != N {
			return false
		}
	}
	return true
}

// EncodeConvLayer encodes ConvLayer to EncodedConvLayer.
func (nn *HENeuralNet) EncodeConvLayer(cl ConvLayer) EncodedConvLayer {
	if !isMatrix(cl.Kernel) {
		panic("kernel not matrix")
	}

	kx := len(cl.Kernel)
	ky := len(cl.Kernel[0])
	windowSize := kx * ky

	// Flatten and repeat kernel
	repeat := ((cl.InputX - kx + cl.Stride) / cl.Stride) * ((cl.InputY - ky + cl.Stride) / cl.Stride)
	flattenedKernel := make([]float64, 0, repeat*windowSize)
	for i := 0; i < kx; i++ {
		for j := 0; j < ky; j++ {
			for n := 0; n < repeat; n++ {
				flattenedKernel = append(flattenedKernel, cl.Kernel[i][j])
			}
		}
	}

	return EncodedConvLayer{
		Im2ColX: windowSize,
		Im2ColY: repeat,

		Kernel: nn.Encoder.EncodeNew(flattenedKernel, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots()),
	}
}

// conv executes ConvLayer in-place.
func (nn *HENeuralNet) conv(cl EncodedConvLayer, ct *rlwe.Ciphertext) {
	// Multiply with kernel
	nn.Evaluator.MulRelin(ct, cl.Kernel, ct)

	// Inner Sum
	nn.Evaluator.InnerSum(ct, cl.Im2ColX, cl.Im2ColY, ct)
}

// EncodeLinearLayer encodes LinearLayer to EncodedLinearLayer.
func (nn *HENeuralNet) EncodeLinearLayer(ll LinearLayer) EncodedLinearLayer {
	if !isMatrix(ll.Weights) {
		panic("weights not matrix")
	}

	N := len(ll.Weights)
	M := len(ll.Weights[0])

	diagWeights := make(map[int][]float64, len(ll.Weights))
	slots := nn.Parameters.Slots()
	for i := 0; i < slots; i++ {
		isZero := true
		row := make([]float64, slots)
		for j := 0; j < slots; j++ {
			ii, jj := j%slots, (i+j)%slots
			if ii < N && jj < M {
				row[j] = ll.Weights[ii][jj]
				if row[j] != 0 {
					isZero = false
				}
			}
			if !isZero {
				diagWeights[i] = row
			}
		}
	}

	encodedWeights := ckks.GenLinearTransform(nn.Encoder, diagWeights, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())
	encodedBias := nn.Encoder.EncodeNew(ll.Bias, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())

	return EncodedLinearLayer{
		Weights: encodedWeights,
		Bias:    encodedBias,
	}
}

// linear executes LinearLayer in-place.
func (nn *HENeuralNet) linear(ll EncodedLinearLayer, ct *rlwe.Ciphertext) {
	cts := []*rlwe.Ciphertext{ct}
	nn.Evaluator.LinearTransform(ct, ll.Weights, cts)
	nn.Evaluator.Add(ct, ll.Bias, ct)
}

// activate executes ActivationLayer in-place.
func (nn *HENeuralNet) activate(l ActivationLayer, ct *rlwe.Ciphertext) {
	l.ActivationFn(ct)
}
