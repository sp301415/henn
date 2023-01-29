package hemnist_test

import (
	"henn"
	"henn/hemnist"
	"testing"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

func BenchmarkInference(b *testing.B) {
	params, _ := ckks.NewParametersFromLiteral(hemnist.DefaultParams)

	var ctx *henn.CKKSContext
	b.Run("CreateContext", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ctx = henn.NewCKKSContext(params)
		}
	})

	var model *henn.HENeuralNet
	b.Run("GenNeuralNet", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model = henn.NewHENeuralNet(params, hemnist.DefaultLayers...)
		}
	})

	b.Run("GenRotationKeys", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			rots := model.Rotations()
			ctx.GenRotationKeys(rots)
		}
	})

	b.Run("InitNeuralNet", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model.Initialize(ctx.EvaluationKey)
		}
	})

	testCase := hemnist.ReadAllTestCase("mnist_test.csv")[0]
	var encImg *rlwe.Ciphertext
	b.Run("EncryptIm2Col", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			encImg = ctx.EncryptIm2Col(testCase.Image, 7, 3)
		}
	})

	b.Run("Inference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model.Infer(encImg)
		}
	})
}

func TestInference(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(hemnist.DefaultParams)
	ctx := henn.NewCKKSContext(params)
	model := henn.NewHENeuralNet(params, hemnist.DefaultLayers...)

	ctx.GenRotationKeys(model.Rotations())

	model.Initialize(ctx.EvaluationKey)

	testSets := hemnist.ReadAllTestCase("mnist_test.csv")
	N := 64
	successes := 0
	for i := 0; i < N; i++ {
		testCase := testSets[i]
		encImg := ctx.EncryptIm2Col(testCase.Image, 7, 3)

		encOutput := model.Infer(encImg)

		output := ctx.DecryptFloats(encOutput, 10)
		pred := hemnist.ArgMax(output)

		t.Logf("Prediction: %v, Label: %v, Success: %v\n", pred, testCase.Label, pred == testCase.Label)
		if pred == testCase.Label {
			successes++
		}
	}
	t.Logf("Accuraccy: %v\n", float64(successes)/float64(N)*100)
}
