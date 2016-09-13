// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
//module Program

open MathNet.Numerics

[<EntryPoint>]
let main argv =
    Control.UseNativeMKL()
//    var net = new Network(new[] { 784, 30, 10 });
//    var data = DataLoader.Load();
//     net.SGD(data.Item1, 30, 10, 1.0, data.Item2);
    0 // return an integer exit code
