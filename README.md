# MLImageLib
Lib with model to classify image

- [NuGet Package](https://www.nuget.org/packages/MLImageLib/)

Easy way to classify image :

    var model = new Model(inceptionTensorFlowModelPath, setsPath, savedModelPath);
    model.FitModel();

    var (score, label) = model.ClassifySingleImage(imagePath);
