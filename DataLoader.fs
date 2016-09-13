module DataLoader

let Load() =
    let trainData = LoadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 50000);
    let testData = LoadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);
    (trainData, testData)


let LoadData imagesFile labelsFile amount =
    let ifsLabels = new FileStream(labelsFileName,   FileMode.Open); // test labels
    let ifsImages = new FileStream(fileName, FileMode.Open); // test images
    let brLabels =  new BinaryReader(ifsLabels);
    let brImages =  new BinaryReader(ifsImages);
    let magic1 = brImages.ReadInt32(); // discard
    let numImages = brImages.ReadInt32();
    let numRows = brImages.ReadInt32();
    let numCols = brImages.ReadInt32();
    let magic2 = brLabels.ReadInt32();
    let numLabels = brLabels.ReadInt32();
    let result = new List<Test>();
    let pixels = new double[784];