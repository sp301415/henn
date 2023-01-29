package hemnist

import (
	_ "embed"
	"henn"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// DefaultLayers are layers that are pre-trained
// using TenSeal tutorial.
var DefaultLayers []henn.Layer

// DefaultParams is parameters optimized for model using DefaultLayers.
var DefaultParams = ckks.ParametersLiteral{
	LogN: 13,
	Q: []uint64{
		0x200038001, // 33
		0x438001, 0x468001, 0x498001,
		0x3e4001, 0x3dc001, 0x3ac001, 0x390001, // 22 * 7
	},
	P:            []uint64{0x20004c001}, // 33
	LogSlots:     12,
	DefaultScale: 1 << 22,
}

func init() {
	activationFn := func(model *henn.HENeuralNet, ct *rlwe.Ciphertext) {
		model.Evaluator.MulRelin(ct, ct, ct)
		model.Evaluator.Rescale(ct, model.Parameters.DefaultScale(), ct)
	}

	DefaultLayers = []henn.Layer{
		henn.ConvLayer{
			InputX: 28,
			InputY: 28,
			Kernel: [][][]float64{convWeight0, convWeight1, convWeight2, convWeight3},
			Bias:   convBias,
			Stride: 3,
		},
		henn.ActivationLayer{
			ActivationFn: activationFn,
		},
		henn.LinearLayer{
			Weights: lin0Weight,
			Bias:    lin0Bias,
		},
		henn.ActivationLayer{
			ActivationFn: activationFn,
		},
		henn.LinearLayer{
			Weights: lin1Weight,
			Bias:    lin1Bias,
		},
	}
}
