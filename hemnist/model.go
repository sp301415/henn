package hemnist

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"henn"
	"strconv"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// DefaultLayers are layers that are pre-trained
// using TenSeal tutorial.
var DefaultLayers []henn.Layer

// DefaultParams is parameter tailer maid from DefaultLayers.
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

var (
	//go:embed model/CW0.csv
	cw0CSV []byte
	//go:embed model/CW1.csv
	cw1CSV []byte
	//go:embed model/CW2.csv
	cw2CSV []byte
	//go:embed model/CW3.csv
	cw3CSV []byte
	//go:embed model/CB.csv
	cbCSV []byte

	//go:embed model/LW1.csv
	lw1CSV []byte
	//go:embed model/LB1.csv
	lb1CSV []byte

	//go:embed model/LW2.csv
	lw2CSV []byte
	//go:embed model/LB2.csv
	lb2CSV []byte
)

func csvToMat(b []byte) [][]float64 {
	csvRd := csv.NewReader(bytes.NewReader(b))
	rows, _ := csvRd.ReadAll()

	M := make([][]float64, len(rows))
	for i, row := range rows {
		M[i] = make([]float64, len(row))
		for j, v := range row {
			M[i][j], _ = strconv.ParseFloat(v, 64)
		}
	}

	return M
}

func init() {
	cw := [][][]float64{
		csvToMat(cw0CSV),
		csvToMat(cw1CSV),
		csvToMat(cw2CSV),
		csvToMat(cw3CSV),
	}
	cb := Flatten(csvToMat(cbCSV))

	lw1 := csvToMat(lw1CSV)
	lb1 := Flatten(csvToMat(lb1CSV))

	lw2 := csvToMat(lw2CSV)
	lb2 := Flatten(csvToMat(lb2CSV))

	activationFn := func(model *henn.HENeuralNet, ct *rlwe.Ciphertext) {
		model.Evaluator.MulRelin(ct, ct, ct)
		model.Evaluator.Rescale(ct, model.Parameters.DefaultScale(), ct)
	}

	DefaultLayers = []henn.Layer{
		henn.ConvLayer{
			InputX: 28,
			InputY: 28,
			Kernel: cw,
			Bias:   cb,
			Stride: 3,
		},
		henn.ActivationLayer{
			ActivationFn: activationFn,
		},
		henn.LinearLayer{
			Weights: lw1,
			Bias:    lb1,
		},
		henn.ActivationLayer{
			ActivationFn: activationFn,
		},
		henn.LinearLayer{
			Weights: lw2,
			Bias:    lb2,
		},
	}
}
