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

// Rotations returns the number of rotations that are needed to infer from given layers.
//
// TODO: This is terribly slow for now, we have to find a way to do this without encoding layers.
func Rotations(params ckks.Parameters, layers []Layer) []int {
	// Create an empty Neural Net, with only necessary information for encoding layers
	nn := &HENeuralNet{
		Parameters: params,
		Encoder:    ckks.NewEncoder(params),
	}

	rotSet := make(map[int]struct{})
	for _, l := range layers {
		switch l := l.(type) {
		case ConvLayer:
			el := nn.EncodeConvLayer(l)

			batchSize := el.Im2ColY
			N := el.Im2ColX

			// Keys needed for InnerSum
			// Taken from InnerSum method in Lattigo
			for i, j := 0, N; j > 0; i, j = i+1, j>>1 {
				if j&1 == 1 {
					k := N - (N & ((2 << i) - 1))
					k *= batchSize
					rotSet[k] = struct{}{}
				}
				rotSet[batchSize*(1<<i)] = struct{}{}
			}

			// Keys needed for masking & addition
			for i := 0; i < len(el.Kernel); i++ {
				rotSet[-i*batchSize] = struct{}{}
			}
		case LinearLayer:
			el := nn.EncodeLinearLayer(l)
			for _, r := range el.Weights.Rotations() {
				rotSet[r] = struct{}{}
			}
		}
	}

	rot := make([]int, 0, len(rotSet))
	for k := range rotSet {
		rot = append(rot, k)
	}
	return rot
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
	for _, k := range cl.Kernel {
		if !isMatrix(k) {
			panic("kernel not matrix")
		}
	}

	if len(cl.Kernel) != len(cl.Bias) {
		panic("dimension mismatch between kernel and bias")
	}

	kx := len(cl.Kernel[0])
	ky := len(cl.Kernel[0][0])
	kSize := kx * ky
	repeat := ((cl.InputX - kx + cl.Stride) / cl.Stride) * ((cl.InputY - ky + cl.Stride) / cl.Stride)
	// imgSize := windowSize * repeat

	encodedKernels := make([]*rlwe.Plaintext, len(cl.Kernel))
	for i, k := range cl.Kernel {
		// Repeat and flatten kernels
		flattenedKernel := make([]float64, 0, repeat*kSize)
		for i := 0; i < kx; i++ {
			for j := 0; j < ky; j++ {
				for n := 0; n < repeat; n++ {
					flattenedKernel = append(flattenedKernel, k[i][j])
				}
			}
		}
		encodedKernels[i] = nn.Encoder.EncodeNew(flattenedKernel, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())
	}

	encodedBiases := make([]*rlwe.Plaintext, len(cl.Bias))
	for i, b := range cl.Bias {
		repeatedBias := make([]float64, kSize)
		for j := range repeatedBias {
			repeatedBias[j] = b
		}
		encodedBiases[i] = nn.Encoder.EncodeNew(repeatedBias, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())
	}

	mask := make([]float64, kSize)
	for i := range mask {
		mask[i] = 1
	}
	encodedMask := nn.Encoder.EncodeNew(mask, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())

	return EncodedConvLayer{
		Im2ColX: kSize,
		Im2ColY: repeat,
		mask:    encodedMask,

		Kernel: encodedKernels,
		Bias:   encodedBiases,
		Stride: cl.Stride,
	}
}

// conv executes ConvLayer in-place.
func (nn *HENeuralNet) conv(cl EncodedConvLayer, ct *rlwe.Ciphertext) {
	// We only have power of two rotation keys,
	// so we make rotation indexes ourselves.

	ctConv := rlwe.NewCiphertext(nn.Parameters.Parameters, ct.Degree(), ct.Level())
	ctTemp := rlwe.NewCiphertext(nn.Parameters.Parameters, ct.Degree(), ct.Level())
	for i := range cl.Kernel {
		k := cl.Kernel[i]
		b := cl.Bias[i]

		// y = k * x + b
		nn.Evaluator.Mul(ct, k, ctTemp)
		nn.Evaluator.InnerSum(ctTemp, cl.Im2ColY, cl.Im2ColX, ctTemp)
		nn.Evaluator.Add(ctTemp, b, ctTemp)

		// Mask and Rotate
		nn.Evaluator.Mul(ctTemp, cl.mask, ctTemp)
		nn.Evaluator.Rotate(ctTemp, -i*cl.Im2ColY, ctTemp)
		nn.Evaluator.Add(ctConv, ctTemp, ctConv)
	}

	ct.Copy(ctConv)
	nn.Evaluator.Rescale(ct, nn.Parameters.DefaultScale(), ct)
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

	encodedWeights := ckks.GenLinearTransformBSGS(nn.Encoder, diagWeights, nn.Parameters.MaxLevel(), nn.Parameters.DefaultScale(), 1.0, nn.Parameters.LogSlots())
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
	l.ActivationFn(nn, ct)
}
