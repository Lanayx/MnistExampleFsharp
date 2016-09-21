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
        let mutable biases = sizes.Tail |>
                             List.map (fun size -> Vector.Build.Random(size, new Normal()))
        let mutable weights = List.zip (List.take(numLayers-1) sizes) sizes.Tail |>
                              List.map (fun (sizeLeft, sizeRight) ->
                              Matrix.Build.Random(sizeRight, sizeLeft, new Normal()))

//        #### Miscellaneous functions
        let Sigmoid(z:Vector<double>) =
//            """The sigmoid function."""
            1.0/(1.0+(-z).PointwiseExp())

        let SigmoidPrime(z:Vector<double>) =
//            """Derivative of the sigmoid function."""
            Sigmoid(z).PointwiseMultiply(1.0-Sigmoid(z));

        let Update_mini_batch (mini_batch: (Vector<double>*Matrix<double>) list) (eta: double) =
//            """Update the network's weights and biases by applying
//            gradient descent using backpropagation to a single mini batch.
//            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
//            is the learning rate."""

            let nabla_b_init = biases |> List.map(fun b -> Vector<double>.Build.Dense(b.Count));
            let nabla_w_init = weights|> List.map(fun w -> Matrix<double>.Build.Dense(w.RowCount,w.ColumnCount));

            let rec calculateNablas (mini_batch: (Vector<double>*Matrix<double>) list) (nabla_b: Vector<double> list) (nabla_w: Matrix<double> list) =
                 match mini_batch with
                 | [] -> (nabla_b, nabla_w)
                 | (x,y)::tail ->
                    let (delta_nabla_b, delta_nabla_w) = Backprop(x, y)
                    let n_b = List.zip nabla_b delta_nabla_b |> List.map(fun (nb, dnb) -> (nb + dnb))
                    let n_w = List.zip nabla_w delta_nabla_w |> List.map(fun (nw, dnw) -> (nw + dnw))
                    calculateNablas tail n_b n_w

            let (nabla_b, nabla_w) = calculateNablas mini_batch nabla_b_init nabla_w_init

            weights <- List.zip weights nabla_w |>
                       List.map(fun (w, nw) -> w-(eta/(double)mini_batch.Length)*nw)
            biases <- List.zip biases nabla_b |>
                      List.map(fun (b, nb) -> b-(eta/(double)mini_batch.Length)*nb)

        let Feedforward(a:Vector<double>) =
//            """Return the output of the network if ``a`` is input."""
            let rec ff l a =
                 match l with
                 | [] -> a
                 | (b,w)::tail -> ff tail (Sigmoid(w*a + b))

            ff (List.zip biases weights) a




        member x.SGD (training_data, epochs, mini_batch_size, eta, test_data) =
            printf "Hello\n"