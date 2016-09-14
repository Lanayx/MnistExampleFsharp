module DataLoader

open System.IO
open MathNet.Numerics.LinearAlgebra.Double

let GetVectorFromNumber lbl =
    let array = Array.create 10 0.0
    array.[lbl] <- (float 1)
    DenseVector.OfArray(array)

let GetRawData imagesCount (pixels: float[]) (brImages: BinaryReader) (brLabels: BinaryReader) = seq {
    for di in 1..imagesCount do
        for i in 0..27 do
            for j in 0..27 do
                pixels.[i*28 + j] <- (float (brImages.ReadByte()))
        let lbl = brLabels.ReadByte();
        yield(DenseVector.OfArray(pixels), GetVectorFromNumber((int lbl)))
}

let LoadData imagesFile labelsFile imagesCount =
    let ifsLabels = new FileStream(labelsFile,   FileMode.Open); // test labels
    let ifsImages = new FileStream(imagesFile, FileMode.Open); // test images
    let brLabels =  new BinaryReader(ifsLabels);
    let brImages =  new BinaryReader(ifsImages);
    let magic1 = brImages.ReadInt32(); // discard
    let numImages = brImages.ReadInt32();
    let numRows = brImages.ReadInt32();
    let numCols = brImages.ReadInt32();
    let magic2 = brLabels.ReadInt32();
    let numLabels = brLabels.ReadInt32();
    let pixels = Array.create 784 0.0
    let data = Seq.toList (GetRawData imagesCount pixels brImages brLabels)

    ifsImages.Close()
    brImages.Close()
    ifsLabels.Close()
    brLabels.Close()
    data


let Load() =
    let trainData = LoadData "train-images.idx3-ubyte" "train-labels.idx1-ubyte" 50000;
    let testData = LoadData "t10k-images.idx3-ubyte" "t10k-labels.idx1-ubyte" 10000;
    (trainData, testData)