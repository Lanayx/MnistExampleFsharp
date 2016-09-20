module Network

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

type Network(sizes: int List) =

        //        "The list ``sizes`` contains the number of neurons in the
        //        respective layers of the network.  For example, if the list
        //        was [2, 3, 1] then it would be a three-layer network, with the
        //        first layer containing 2 neurons, the second layer 3 neurons,
        //        and the third layer 1 neuron.  The biases and weights for the
        //        network are initialized randomly, using a Gaussian
        //        distribution with mean 0, and variance 1.  Note that the first
        //        layer is assumed to be an input layer, and by convention we
        //        won't set any biases for those neurons, since biases are only
        //        ever used in computing the outputs from later layers."
        let numLayers = sizes.Length
        let sizes = sizes
        let biases = sizes.Tail |>
                     List.map (fun size -> Vector.Build.Random(size, new Normal()))
        let weights = List.zip (List.take(numLayers-1) sizes) sizes.Tail |>
                      List.map (fun (sizeLeft, sizeRight) ->
                      Matrix.Build.Random(sizeRight, sizeLeft, new Normal()))

//        #### Miscellaneous functions
        let Sigmoid(z:Vector<double>) =
//            """The sigmoid function."""
            1.0/(1.0+(-z).PointwiseExp())

        let SigmoidPrime(z:Vector<double>) =
//            """Derivative of the sigmoid function."""
            Sigmoid(z).PointwiseMultiply(1- Sigmoid(z));



        let feedforward(a:Vector<double>) =
//            """Return the output of the network if ``a`` is input."""           
            let rec ff l a =
                 match l with
                 | [] -> a
                 | (b,w)::tail -> ff tail (Sigmoid(w*a + b))

            ff (List.zip biases weights) a




        member x.SGD (training_data, epochs, mini_batch_size, eta, test_data) =
            printf "Hello\n"