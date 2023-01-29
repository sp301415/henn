package henn

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// HENeuralNet represents the Neural Network with Homomorphic Encryption Operations.
type HENeuralNet struct {
	Parameters    ckks.Parameters
	EvaluationKey rlwe.EvaluationKey
	Encoder       ckks.Encoder
	Evaluator     ckks.Evaluator
	Layers        []EncodedLayer
}

// NewHENeuralNet returns the empty HENeuralNet with Encoder initialized.
// To use this NN, you should call initialize with PubicKeySet.
func NewHENeuralNet(params ckks.Parameters, layers ...Layer) *HENeuralNet {
	nn := &HENeuralNet{
		Parameters: params,
		Encoder:    ckks.NewEncoder(params),
		Evaluator:  nil,
	}
	nn.AddLayers(layers...)

	return nn
}

// Initialize intializes this neural network using sender's public evaluation keys.
func (nn *HENeuralNet) Initialize(evk rlwe.EvaluationKey) {
	nn.EvaluationKey = evk
	nn.Evaluator = ckks.NewEvaluator(nn.Parameters, evk)
}

// Rotations returns the number of rotations that are needed to infer from this neural network.
func (nn *HENeuralNet) Rotations() []int {
	rotSet := make(map[int]struct{})

	for _, l := range nn.Layers {
		switch l := l.(type) {
		case EncodedConvLayer:
			batchSize := l.Im2ColY
			N := l.Im2ColX

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
			for i := 0; i < len(l.Kernel); i++ {
				rotSet[-i*batchSize] = struct{}{}
			}

		case EncodedLinearLayer:
			for _, r := range l.Weights.Rotations() {
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
func (nn *HENeuralNet) Infer(ctIn *rlwe.Ciphertext) *rlwe.Ciphertext {
	if nn.Evaluator == nil {
		panic("model not initialized")
	}

	ctOut := ctIn.CopyNew()

	for _, l := range nn.Layers {
		switch l := l.(type) {
		case EncodedConvLayer:
			nn.conv(l, ctOut)
		case EncodedLinearLayer:
			nn.linear(l, ctOut)
		case ActivationLayer:
			nn.activate(l, ctOut)
		}
	}

	return ctOut
}

// EncodeConvLayer encodes ConvLayer to EncodedConvLayer.
func (nn *HENeuralNet) EncodeConvLayer(cl ConvLayer) EncodedConvLayer {
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
	nn.Evaluator.LinearTransform(ct, ll.Weights, []*rlwe.Ciphertext{ct})
	nn.Evaluator.Rescale(ct, nn.Parameters.DefaultScale(), ct)
	nn.Evaluator.Add(ct, ll.Bias, ct)
}

// activate executes ActivationLayer in-place.
func (nn *HENeuralNet) activate(l ActivationLayer, ct *rlwe.Ciphertext) {
	l.ActivationFn(nn, ct)
}
