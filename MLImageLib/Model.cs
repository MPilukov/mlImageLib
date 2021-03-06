using MLImageLib.Contracts;
using Microsoft.ML;
using System.Linq;

namespace MLImageLib
{
    public class Model
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private DataViewSchema _schema;
        private PredictionEngine<ImageData, ImagePrediction> _predictor;

        private readonly string _inceptionTensorFlowModel; // inception path
        private readonly string _savedModelPath; // path to store model
        private readonly string _setsPath; // sets path

        /// <summary>
        /// Ctr Model
        /// </summary>
        /// <param name="inceptionTensorFlowModel">Inception path</param>
        /// <param name="setsPath">Sets path</param>
        /// <param name="savedModelPath">Path to store model</param>
        public Model(string inceptionTensorFlowModel, string setsPath, string savedModelPath)
        {
            _mlContext = new MLContext();
            _inceptionTensorFlowModel = inceptionTensorFlowModel;

            _setsPath = setsPath;
            _savedModelPath = savedModelPath;
        }

        public void FitModel()
        {
            TrainModel(_setsPath, _inceptionTensorFlowModel);
            SaveModel();
        }

        /// <summary>
        /// Get image classification
        /// </summary>
        /// <param name="filePath">Image path</param>
        /// <returns></returns>
        public (float[] score, string label) ClassifySingleImage(string filePath)
        {
            if (_model == null)
                LoadModel();

            if (_predictor == null)
                _predictor = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(_model);

            var imageData = new ImageData
            {
                ImagePath = filePath
            };

            var response = _predictor.Predict(imageData);
            return (response.Score, response.PredictedLabelValue);
        }

        private void LoadModel()
        {
            _model = _mlContext.Model.Load(_savedModelPath, out _schema);
        }

        private void SaveModel()
        {
            _mlContext.Model.Save(_model, _schema, _savedModelPath);
        }

        private double TrainModel(string setsPath, string inceptionTensorFlowModel)
        {
            var pipeline = _mlContext.Transforms
                .LoadImages
                (
                    outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath)
                )
                .Append(_mlContext.Transforms
                .ResizeImages(
                    outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth,
                    imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input")
                )
                .Append(_mlContext.Transforms
                .ExtractPixels(
                    outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast,
                    offsetImage: InceptionSettings.Mean)
                )
                .Append(_mlContext.Model.LoadTensorFlowModel(inceptionTensorFlowModel).
                    ScoreTensorFlowModel(outputColumnNames:
                    new[] { "softmax2_pre_activation" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true))
                .Append(_mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers
                .LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(_mlContext);

            var (train, test) = ImageData.ReadData(setsPath);
            var trainingData = _mlContext.Data.LoadFromEnumerable<ImageData>(train);
            _model = pipeline.Fit(trainingData);
            var testData = _mlContext.Data.LoadFromEnumerable<ImageData>(test);
            var predictions = _model.Transform(testData);
            var imagePredictionData =
                _mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true).ToList();
            var metrics =
                _mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");
            _schema = trainingData.Schema;
            return metrics.LogLoss;
        }
    }
}