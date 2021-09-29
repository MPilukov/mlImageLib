using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLImageLib.Contracts
{
    internal class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static (IEnumerable<ImageData> train, IEnumerable<ImageData> test) ReadData(string pathToFolder)
        {
            var list = new List<ImageData>();
            var directories = Directory.EnumerateDirectories(pathToFolder);

            foreach (var dir in directories)
            {
                var label = (new DirectoryInfo(dir)).Name;
                
                foreach (var file in Directory.GetFiles(dir))
                {
                    list.Add(new ImageData
                    {
                        ImagePath = file,
                        Label = label,
                    });
                }
            }

            list = list.Shuffle().ToList();
            return GetSets(list);
        }

        public static (IEnumerable<ImageData> train, IEnumerable<ImageData> test) GetSets(List<ImageData> data)
        {
            var trainCount = data.Count * 99 / 100;
            var train = data.Take(trainCount);
            var test = data.Skip(trainCount);
            return (train, test);
        }
    }
}