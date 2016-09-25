module Network

open System
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
        let mutable backPropTime = 0.0
        let mutable evalTime = 0.0
        let mutable ubmExtras = 0.0
        let mutable nbwTime = 0.0
//        #### Miscellaneous functions
        let Sigmoid(z:Vector<double>) =
//            """The sigmoid function."""
            1.0/(1.0+(-z).PointwiseExp())

        let SigmoidPrime(z:Vector<double>) =
//            """Derivative of the sigmoid function."""
            Sigmoid(z).PointwiseMultiply(1.0-Sigmoid(z));

        let CostDerivative (output_activations:Vector<double>) (y:Vector<double>) =
//            """Return the vector of partial derivatives \partial C_x /
//            \partial a for the output activations."""
            output_activations-y

        let Backprop (x:Vector<double>) (y:Vector<double>) =
//            """Return a tuple ``(nabla_b, nabla_w)`` representing the
//            gradient for the cost function C_x.  ``nabla_b`` and
//            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
//            to ``self.biases`` and ``self.weights``."""

//            let nabla_b_init = [for b in biases -> Vector<double>.Build.Dense(b.Count)]
//            let nabla_w_init = [for w in weights -> Matrix<double>.Build.Dense(w.RowCount,w.ColumnCount)]

//            # feedforward
            let bpStart = DateTime.Now
            let rec ffstart (biasesAndWeights: (Vector<double>*Matrix<double>) list) (zs: Vector<double> list)  (activations: Vector<double> list) (activation : Vector<double>) =
                match biasesAndWeights with
                | [] -> (zs, activations)
                | (b,w)::tail ->
                    let z = w*activation+b
                    let a = Sigmoid(z)
                    ffstart tail (z::zs) (a::activations) a

            let biasesAndWeights = List.zip biases weights
            let zs, activations = ffstart biasesAndWeights [] [x] x  //zs and activations are reverted

            //# backward pass
            let delta_init = (CostDerivative activations.Head y).PointwiseMultiply(SigmoidPrime(zs.Head))
            let nabla_b_init = [ delta_init ]
            let nabla_w_init = [ delta_init.ToColumnMatrix() * activations.Tail.Head.ToRowMatrix()];

            let zsActivationsAndWeights = List.zip3 (zs.Tail) (activations.Tail.Tail) (List.rev weights.Tail)

            let rec ffend (zaw: (Vector<double>*Vector<double>*Matrix<double>) list) (delta : Vector<double>) nb nw =
                match zaw with
                | [] -> (nb, nw)
                | (z, a, w)::tail ->
                    let sp = SigmoidPrime(z)
                    let deltaNew = (w.Transpose()*delta).PointwiseMultiply(sp)
                    let nabla_b = deltaNew
                    let nabla_w = deltaNew.ToColumnMatrix() *a.ToRowMatrix()
                    ffend tail deltaNew (nabla_b::nb) (nabla_w::nw)

            let result = ffend zsActivationsAndWeights delta_init nabla_b_init nabla_w_init
            backPropTime <- backPropTime + (DateTime.Now - bpStart).TotalSeconds
            result

        let Update_mini_batch (mini_batch: (Vector<double>*Vector<double>) list) (eta: double) =
//            """Update the network's weights and biases by applying
//            gradient descent using backpropagation to a single mini batch.
//            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
//            is the learning rate."""
            let ubmInit = DateTime.Now
            let nabla_b_init = biases |> List.map(fun b -> Vector<double>.Build.Dense(b.Count));
            let nabla_w_init = weights|> List.map(fun w -> Matrix<double>.Build.Dense(w.RowCount,w.ColumnCount));
            ubmExtras <- ubmExtras + (DateTime.Now - ubmInit).TotalSeconds

            let rec calculateNablas (mini_batch: (Vector<double>*Vector<double>) list) (nabla_b: Vector<double> list) (nabla_w: Matrix<double> list) =
                 match mini_batch with
                 | [] -> (nabla_b, nabla_w)
                 | (x,y)::tail ->
                    let (delta_nabla_b, delta_nabla_w) = Backprop x y
                    let nbw = DateTime.Now
                    let n_b = [ for (nb, dnb) in  List.zip nabla_b delta_nabla_b -> (nb + dnb)]
                    let n_w = [ for (nw, dnw) in  List.zip nabla_w delta_nabla_w -> (nw + dnw)]
                    nbwTime <- nbwTime + (DateTime.Now - nbw).TotalSeconds
                    calculateNablas tail n_b n_w

            let (nabla_b, nabla_w) = calculateNablas mini_batch nabla_b_init nabla_w_init

            let wbCals = DateTime.Now
            weights <- [ for (w, nw) in List.zip weights nabla_w -> w-(eta/(double)mini_batch.Length)*nw ]
            biases <- [ for (b, nb) in  List.zip biases nabla_b -> b-(eta/(double)mini_batch.Length)*nb ]
            ubmExtras <- ubmExtras + (DateTime.Now - wbCals).TotalSeconds

        let Feedforward(a:Vector<double>) =
//            """Return the output of the network if ``a`` is input."""
            let rec ff l a =
                 match l with
                 | [] -> a
                 | (b,w)::tail -> ff tail (Sigmoid(w*a + b))

            ff (List.zip biases weights) a


        let Evaluate(test_data: (Vector<double>*Vector<double>) []) =
             let evStart = DateTime.Now
    //        """Return the number of test inputs for which the neural
    //        network outputs the correct result. Note that the neural
    //        network's output is assumed to be the index of whichever
    //        neuron in the final layer has the highest activation."""
             let test_results = [for (x, y) in test_data -> (Feedforward(x).MaximumIndex(), y.MaximumIndex())]
                                |> List.map(fun (i1,i2) -> if i1 = i2 then 1 else 0 )
             let result = List.sum test_results
             evalTime <- evalTime + (DateTime.Now - evStart).TotalSeconds
             result

        // shuffle an array (in-place)
        let shuffle a (rand:Random) =
            let swap (a: _[]) x y =
                let tmp = a.[x]
                a.[x] <- a.[y]
                a.[y] <- tmp
            Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a
            Array.toList a

        member x.SGD (training_data: _ [], epochs, mini_batch_size, eta, test_data: _ []) =
//            """Train the neural network using mini-batch stochastic
//            gradient descent.  The ``training_data`` is a list of tuples
//            ``(x, y)`` representing the training inputs and the desired
//            outputs.  The other non-optional parameters are
//            self-explanatory.  If ``test_data`` is provided then the
//            network will be evaluated against the test data after each
//            epoch, and partial progress printed out.  This is useful for
//            tracking progress, but slows things down substantially."""

            let rnd = new Random();
            let startTime = DateTime.Now;

            let n_test = test_data.Length
            let n = training_data.Length
            for j in [1..epochs] do
                let shuffledTrainingData = shuffle training_data rnd
                let mini_batches = List.chunkBySize mini_batch_size shuffledTrainingData
                for mini_batch in mini_batches do
                    Update_mini_batch mini_batch eta
                printf "Epoch %d: %d / %d (%f) " j (Evaluate(test_data)) n_test (DateTime.Now - startTime).TotalSeconds
                printf "BP: %f EV: %f UBME:%f NBW: %f \n" backPropTime evalTime ubmExtras nbwTime


