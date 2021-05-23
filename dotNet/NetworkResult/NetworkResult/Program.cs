using NetworkResult.Helpers;
using NetworkResult.Models;
using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace NetworkResult
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var workingDataset = Dataset.UcLanduse;
            var dirPath = $"Data\\{workingDataset.Name}\\";
            string[] fileEntries = Directory.GetFiles(dirPath);

            var allTrainStats = new List<TrainStats>();

            foreach (string fileName in fileEntries)
            {
                try
                {
                    var trainStats = await ParseToObject(fileName);
                    allTrainStats.Add(trainStats);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }



            var averageResults = new List<AverageResult>();

            foreach (var model in Helper.Models)
            {
                var bestEpoch = new Dictionary<string, List<int>>();
                var totalTime = new Dictionary<string, List<float>>();
                var trainTime = new Dictionary<string, List<float>>();
                var evalTime = new Dictionary<string, List<float>>();
                var accuracy = new Dictionary<string, List<float>>();
                var k1 = new Dictionary<string, List<float>>();
                var k5 = new Dictionary<string, List<float>>();

                var stat = allTrainStats.Where(x => x.ModelName == model).ToList();
                bestEpoch.Add(model, stat.Select(s => s.BestEpoch).ToList());
                totalTime.Add(model, stat.Select(s => s.TotalTime).ToList());
                trainTime.Add(model, stat.Select(s => s.TrainTime).ToList());
                evalTime.Add(model, stat.Select(s => s.EvalTime).ToList());
                accuracy.Add(model, stat.Select(s => s.Accuracy).ToList());
                k1.Add(model, stat.Select(s => s.K1).ToList());
                k5.Add(model, stat.Select(s => s.K5).ToList());


                var avgBestEpoch = bestEpoch.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgTotalTime = totalTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgTrainTime = trainTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgEvalTime = evalTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgAccuracy = accuracy.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgK1 = k1.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgK5 = k5.Where(t => t.Key == model).Average(t => t.Value.Average());

                var minBestEpoch = bestEpoch.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minTotalTime = totalTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minTrainTime = trainTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minEvalTime = evalTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minAccuracy = accuracy.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minK1 = k1.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minK5 = k5.Where(t => t.Key == model).Min(t => t.Value.Min());

                var maxBestEpoch = bestEpoch.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxTotalTime = totalTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxTrainTime = trainTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxEvalTime = evalTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxAccuracy = accuracy.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxK1 = k1.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxK5 = k5.Where(t => t.Key == model).Max(t => t.Value.Max());


                var TrainAccuracy = stat.Select(s => s.TrainAccuracy).ToList();
                var ValidAccuracy = stat.Select(s => s.ValidAccuracy).ToList();
                var TrainLoss = stat.Select(s => s.TrainLoss).ToList();
                var ValidLoss = stat.Select(s => s.ValidLoss).ToList();


                MergeArrays(ValidAccuracy);
                MergeArrays(TrainLoss);
                MergeArrays(ValidLoss);


                (var MinTrainAccuracy, var MaxTrainAccuracy, var AvgTrainAccuracy) = MergeArrays(TrainAccuracy);
                (var MinTrainLoss, var MaxTrainLoss, var AvgTrainLoss) = MergeArrays(TrainLoss);
                (var MinValidAccuracy, var MaxValidAccuracy, var AvgValidAccuracy) = MergeArrays(ValidAccuracy);
                (var MinValidLoss, var MaxValidLoss, var AvgValidLoss) = MergeArrays(ValidLoss);



                averageResults.Add(new AverageResult
                {
                    ModelName = model,
                    AvgAccuracy = avgAccuracy,
                    AvgBestEpoch = (int)Math.Floor(avgBestEpoch),
                    AvgEvalTime = avgEvalTime,
                    AvgTotalTime = avgTotalTime,
                    AvgTrainTime = avgTrainTime,
                    AvgK1 = avgK1,
                    AvgK5 = avgK5,
                    MaxBestEpoch = maxBestEpoch,
                    MaxAccuracy = maxAccuracy,
                    MaxEvalTime = maxEvalTime,
                    MaxTotalTime = maxTotalTime,
                    MaxTrainTime = maxTrainTime,
                    MaxK1 = maxK1,
                    MaxK5 = maxK5,
                    MinBestEpoch = minBestEpoch,
                    MinAccuracy = minAccuracy,
                    MinEvalTime = minEvalTime,
                    MinTotalTime = minTotalTime,
                    MinTrainTime = minTrainTime,
                    MinK1 = minK1,
                    MinK5 = minK5,
                    MinTrainAccuracy = MinTrainAccuracy,
                    MaxTrainAccuracy = MaxTrainAccuracy,
                    AvgTrainAccuracy = AvgTrainAccuracy,
                    MinTrainLoss = MinTrainLoss,
                    MaxTrainLoss = MaxTrainLoss,
                    AvgTrainLoss = AvgTrainLoss,
                    MinValidAccuracy = MinValidAccuracy,
                    MaxValidAccuracy = MaxValidAccuracy,
                    AvgValidAccuracy = AvgValidAccuracy,
                    MinValidLoss = MinValidLoss,
                    MaxValidLoss = MaxValidLoss,
                    AvgValidLoss = AvgValidLoss,
                });
            }

            await SaveToJsonFile(averageResults, "averageResults.json");

            Console.WriteLine("Done!");
            Console.ReadKey();

        }

        static async Task<TrainStats> ParseToObject(string filePath)
        {
            using FileStream openStream = File.OpenRead(filePath);
            TrainStats trainStats = await JsonSerializer.DeserializeAsync<TrainStats>(openStream);
            return trainStats;
        }

        static async Task SaveToJsonFile(List<AverageResult> averageResults, string fileName)
        {
            using FileStream createStream = File.Create(fileName);
            await JsonSerializer.SerializeAsync(createStream, averageResults);
        }

        static (List<float>, List<float>, List<float>) MergeArrays(List<List<float>> arrays)
        {
            var largest = arrays.Max(a => a.Count);
            var minArr = new List<float>();
            var maxArr = new List<float>();
            var avgArr = new List<float>();


            for (int i = 0; i < arrays.Count; i++)
            {
                for (int j = 0; j < largest; j++)
                {
                    if (arrays[i].Count <= j)
                    {
                        break;
                    }

                    if (minArr.Count < j+1)
                    {
                        maxArr.Add(arrays[i][j]);
                        minArr.Add(arrays[i][j]);
                        avgArr.Add(arrays[i][j] / arrays.Where(a => a.Count >= j).Count());
                    }
                    else
                    {
                        minArr[j] = arrays[i][j] < minArr[j] ? arrays[i][j] : minArr[j];
                        maxArr[j] = arrays[i][j] > maxArr[j] ? arrays[i][j] : maxArr[j];
                        avgArr[j] += arrays[i][j] / arrays.Where(a => a.Count >= j).Count();
                    }
                }
            }


            return (minArr, maxArr, avgArr);
        }
    }

}
