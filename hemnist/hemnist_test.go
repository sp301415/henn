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

	b.Run("GenRotationKeys", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			rots := henn.Rotations(params, hemnist.DefaultLayers)
			ctx.GenRotationKeys(rots)
		}
	})

	testCase := hemnist.ReadAllTestCase("mnist_test.csv")[0]
	var encImg *rlwe.Ciphertext
	b.Run("EncryptIm2Col", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			encImg = ctx.EncryptIm2Col(testCase.Image, 7, 3)
		}
	})

	pks := ctx.PublicKeySet()
	var model *henn.HENeuralNet
	b.Run("GenNeuralNet", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model = henn.NewHENeuralNet(pks)
			model.AddLayers(hemnist.DefaultLayers...)
		}
	})

	b.Run("Inference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			model.Infer(encImg)
		}
	})
}
