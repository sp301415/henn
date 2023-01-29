package main

import (
	"fmt"
	"henn"
	"henn/hemnist"
	"image"
	_ "image/jpeg"
	"os"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	// This is a sample program to simulate server-client MNIST inference using Homomorphic Encryption.

	// Choose parameters to share between client and server.
	// Our model requires depth 6, so we have to choose large enough parameter.
	params, _ := ckks.NewParametersFromLiteral(hemnist.DefaultParams)

	// Client creates CKKSContext.
	ctx := henn.NewCKKSContext(params)

	// Server initializes the model using pre-trained layers.
	model := henn.NewHENeuralNet(params, hemnist.DefaultLayers...)

	// Using model information, we calculate the rotation indexes needed
	// and create rotation keys.
	rots := model.Rotations()
	ctx.GenRotationKeys(rots)

	// Client sends evaluation keys to server.
	// Server uses this to initialize the model.
	evk := ctx.EvaluationKey
	model.Initialize(evk)

	// Client encrypts the image using Im2Col, and sends it to server.
	f, err := os.Open("9.jpg")
	if err != nil {
		panic(err)
	}
	img, _, _ := image.Decode(f)

	testCase := hemnist.NormalizeImage(img)
	encImg := ctx.EncryptIm2Col(testCase, 7, 3) // Our model uses 7*7 Kernel with Stride 3.

	// Server calculates the inferred result, and sends it to client.
	encOutput := model.Infer(encImg)

	// Client decrypts the result from the server, and obtains the result.
	output := ctx.DecryptFloats(encOutput, 10) // 0 ~ 9
	pred := hemnist.ArgMax(output)

	fmt.Println("Prediction:", pred)
	fmt.Println("Raw Output:", output)
}
