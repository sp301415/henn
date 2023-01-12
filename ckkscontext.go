package henn

import (
	"math"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// CKKSContext contains various structures for CKKS operations for clients,
// such as Encoder, Encryptor, Decryptor, Evaluator, and KeyGenerator.
type CKKSContext struct {
	Parameters ckks.Parameters

	Encoder   ckks.Encoder
	Encryptor rlwe.Encryptor
	Decryptor rlwe.Decryptor
	Evaluator ckks.Evaluator

	KeyGenerator  rlwe.KeyGenerator
	PublicKey     *rlwe.PublicKey
	SecretKey     *rlwe.SecretKey
	EvaluationKey rlwe.EvaluationKey
}

// PublicKeySet is a structure containing public information about certain CKKSContext.
// Use this to initialize HENeuralNet.
type PublicKeySet struct {
	Parameters    ckks.Parameters
	PublicKey     *rlwe.PublicKey
	EvaluationKey rlwe.EvaluationKey
}

// NewCKKSContext creates a new CKKSContext.
// This DOES NOT create rotation keys. Use GenRotationKeys instaed.
func NewCKKSContext(params ckks.Parameters) *CKKSContext {
	keyGenerator := ckks.NewKeyGenerator(params)
	sk, pk := keyGenerator.GenKeyPair()
	rlk := keyGenerator.GenRelinearizationKey(sk, 2)
	evk := rlwe.EvaluationKey{Rlk: rlk}

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, evk)

	return &CKKSContext{
		Parameters: params,

		Encoder:   encoder,
		Encryptor: encryptor,
		Decryptor: decryptor,
		Evaluator: evaluator,

		KeyGenerator:  keyGenerator,
		PublicKey:     pk,
		SecretKey:     sk,
		EvaluationKey: evk,
	}
}

// GenRotationKeys creates rotation keys and stores them internally.
func (ctx *CKKSContext) GenRotationKeys(rots []int) {
	rtks := ctx.KeyGenerator.GenRotationKeysForRotations(rots, false, ctx.SecretKey)
	ctx.EvaluationKey = rlwe.EvaluationKey{Rlk: ctx.EvaluationKey.Rlk, Rtks: rtks}
	ctx.Evaluator = ckks.NewEvaluator(ctx.Parameters, ctx.EvaluationKey)
}

// PublicKeySet returns the PublicKeySet of this context.
func (ctx *CKKSContext) PublicKeySet() *PublicKeySet {
	return &PublicKeySet{
		Parameters:    ctx.Parameters,
		PublicKey:     ctx.PublicKey,
		EvaluationKey: ctx.EvaluationKey,
	}
}

// EncryptInts encodes and encrypts slices of int to ckks ciphertext.
func (ctx *CKKSContext) EncryptInts(msg []int) *rlwe.Ciphertext {
	msgFloats := make([]float64, len(msg))
	for i, v := range msg {
		msgFloats[i] = float64(v)
	}
	return ctx.EncryptFloats(msgFloats)
}

// EncryptFloats encodes and encrypts slices of float64s to ckks ciphertext.
func (ctx *CKKSContext) EncryptFloats(msg []float64) *rlwe.Ciphertext {
	pt := ctx.Encoder.EncodeNew(msg, ctx.Parameters.MaxLevel(), ctx.Parameters.DefaultScale(), ctx.Parameters.LogSlots())
	return ctx.Encryptor.EncryptNew(pt)
}

// DecryptInts decrypts and decodes encryption of slices of ints.
func (ctx *CKKSContext) DecryptInts(ct *rlwe.Ciphertext, len int) []int {
	pt := ctx.Decryptor.DecryptNew(ct)
	msgCmplx := ctx.Encoder.Decode(pt, ctx.Parameters.LogSlots())
	msg := make([]int, len)
	for i := range msg {
		msg[i] = int(math.Round(real(msgCmplx[i])))
	}
	return msg
}

// DecryptFloats decrypts and decodes encryption of slices of float64s.
func (ctx *CKKSContext) DecryptFloats(ct *rlwe.Ciphertext, len int) []float64 {
	pt := ctx.Decryptor.DecryptNew(ct)
	msgCmplx := ctx.Encoder.Decode(pt, ctx.Parameters.LogSlots())
	msg := make([]float64, len)
	for i := range msg {
		msg[i] = real(msgCmplx[i])
	}
	return msg
}

// EncryptIm2Col encrypts an image(2D slice) as column form,
// which enables convolution with kernels.
//
// Refer to TenSeal paper for more information.
func (ctx *CKKSContext) EncryptIm2Col(img [][]float64, kernelSize int, stride int) *rlwe.Ciphertext {
	X := len(img)
	Y := len(img[0])

	if (X-kernelSize+stride)%stride != 0 || (Y-kernelSize+stride)%stride != 0 {
		panic("size mismatch")
	}

	XX := kernelSize * kernelSize
	YY := ((X - kernelSize + stride) / stride) * ((Y - kernelSize + stride) / stride)

	encodedImg := make([][]float64, XX)
	for i := range encodedImg {
		encodedImg[i] = make([]float64, YY)
	}

	// Im2Col
	var xx, yy int
	for i := 0; i <= X-kernelSize; i += stride {
		for j := 0; j <= Y-kernelSize; j += stride {
			for ki := 0; ki < kernelSize; ki++ {
				for kj := 0; kj < kernelSize; kj++ {
					encodedImg[xx][yy] = img[i+ki][j+kj]
					xx++
				}
			}
			xx, yy = 0, yy+1
		}
	}

	// Flatten by vertical scanning
	// NOTE: It's already vertically aligned after Im2Col,
	// so we can just append everything
	flattened := make([]float64, 0, XX*YY)
	for _, row := range encodedImg {
		flattened = append(flattened, row...)
	}
	return ctx.EncryptFloats(flattened)
}
