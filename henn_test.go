package henn

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

var ctx *CKKSContext

func init() {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	ctx = NewCKKSContext(params)
}

func TestEncryptDecrypt(t *testing.T) {
	r := rand.New(rand.NewSource(0))
	N := 32

	t.Run("Ints", func(t *testing.T) {
		pt := make([]int, N)
		for i := range pt {
			pt[i] = r.Intn(64) // Too large integers can have errors embedded!
		}
		pt2 := ctx.DecryptInts(ctx.EncryptInts(pt), N)
		if !reflect.DeepEqual(pt, pt2) {
			t.Fail()
		}
	})

	t.Run("Floats", func(t *testing.T) {
		pt := make([]float64, N)
		for i := range pt {
			pt[i] = r.Float64()
		}
		pt2 := ctx.DecryptFloats(ctx.EncryptFloats(pt), N)
		for i := 0; i < N; i++ {
			if math.Abs(pt[i]-pt2[i]) > 1e-3 {
				t.Fail()
			}
		}
	})
}

func TestIm2Col(t *testing.T) {
	t.Run("4*2/2/2", func(t *testing.T) {
		// Image:
		// 1 2 3 4
		// 5 6 7 8
		// Kernel Size: 2 * 2
		// Stride: 2
		//
		// After Im2Col:
		// 1 2 5 6
		// 3 4 7 8
		//
		// After vertical scanning:
		// 1 3 2 4 5 7 6 8

		img := [][]float64{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		}
		ct := ctx.EncryptIm2Col(img, 2, 2)
		pt := ctx.DecryptInts(ct, 8)

		if !reflect.DeepEqual(pt, []int{1, 3, 2, 4, 5, 7, 6, 8}) {
			t.Fail()
		}
	})

	t.Run("3*3/2/1", func(t *testing.T) {
		// This is from the official example of TenSeal.

		img := [][]float64{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		}
		ct := ctx.EncryptIm2Col(img, 2, 1)
		pt := ctx.DecryptInts(ct, 16)

		if !reflect.DeepEqual(pt, []int{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9}) {
			t.Fail()
		}
	})
}

func TestConv(t *testing.T) {
	img := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	kernel := [][]float64{
		{1, 1},
		{1, 1},
	}
	stride := 1
	convLayer := ConvLayer{
		InputX: len(img),
		InputY: len(img[0]),
		Kernel: [][][]float64{kernel, kernel},
		Bias:   []float64{0, 1},
		Stride: stride,
	}

	ct := ctx.EncryptIm2Col(img, len(kernel), stride)
	ctx.GenRotationKeys(Rotations(ctx.Parameters, []Layer{convLayer}))

	nn := NewHENeuralNet(ctx.PublicKeySet())
	nn.AddLayers(convLayer)
	ct = nn.Infer(ct)

	pt := ctx.DecryptInts(ct, 8)

	if !reflect.DeepEqual(pt, []int{12, 16, 24, 28, 13, 17, 25, 29}) {
		t.Fail()
	}
}

func TestLinear(t *testing.T) {
	linearLayer := LinearLayer{
		Weights: [][]float64{
			{1, 2},
			{3, 4},
			{5, 6},
		},
		Bias: []float64{2, 0, 0},
	}
	ctx.GenRotationKeys(Rotations(ctx.Parameters, []Layer{linearLayer}))

	nn := NewHENeuralNet(ctx.PublicKeySet())
	nn.AddLayers(linearLayer)

	ct := ctx.EncryptInts([]int{1, 1})
	ct = nn.Infer(ct)
	pt := ctx.DecryptInts(ct, 3)

	if !reflect.DeepEqual(pt, []int{5, 7, 11}) {
		t.Fail()
	}
}
