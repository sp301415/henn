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

	ct := ctx.EncryptInts([]int{1, 2, 3, 4})
	ctx.Evaluator.Rotate(ct, 1, ct)
	fmt.Println(ctx.DecryptInts(ct, 4))

}
