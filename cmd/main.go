package main

import (
	"fmt"
	"henn"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func make2d[T any](x, y int) [][]T {
	res := make([][]T, x)
	for i := range res {
		res[i] = make([]T, y)
	}
	return res
}

func main() {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN12QP109)
	ctx := henn.NewCKKSContext(params)

	img := make2d[int](3, 3)
	for i := 0; i < 9; i++ {
		img[i/3][i%3] = i + 1
	}
	fmt.Println(img)
	ct := ctx.EncryptIm2Col(img, 2, 1)
	fmt.Println(ctx.DecryptInts(ct, 9))

}
