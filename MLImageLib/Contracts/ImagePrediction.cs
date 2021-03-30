using Microsoft.ML.Data;

namespace MLImageLib.Contracts
{
    internal class ImagePrediction : ImageData
    {
        [ColumnName("Score")]
        public float[] Score;

        public string PredictedLabelValue;
    }
}