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
            var workingDataset = Dataset.Resisc;
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

            foreach (var model in Helper.ResiscModels)
            {
                var bestEpoch = new Dictionary<string, List<int>>();
                var totalTime = new Dictionary<string, List<float>>();
                var trainTime = new Dictionary<string, List<float>>();
                var evalTime = new Dictionary<string, List<float>>();
                var accuracy = new Dictionary<string, List<float>>();
                var k1 = new Dictionary<string, List<float>>();
                var k5 = new Dictionary<string, List<float>>();
                var fMeasure = new Dictionary<string, List<float>>();
                var precision = new Dictionary<string, List<float>>();
                var recall = new Dictionary<string, List<float>>();

                var stat = allTrainStats.Where(x => x.ModelName == model).ToList();
                bestEpoch.Add(model, stat.Select(s => s.BestEpoch).ToList());
                totalTime.Add(model, stat.Select(s => s.TotalTime).ToList());
                trainTime.Add(model, stat.Select(s => s.TrainTime).ToList());
                evalTime.Add(model, stat.Select(s => s.EvalTime).ToList());
                accuracy.Add(model, stat.Select(s => s.Accuracy).ToList());
                k1.Add(model, stat.Select(s => s.K1).ToList());
                k5.Add(model, stat.Select(s => s.K5).ToList());
                fMeasure.Add(model, stat.Select(s => s.FMeasure).ToList());
                precision.Add(model, stat.Select(s => s.Precision).ToList());
                recall.Add(model, stat.Select(s => s.Recall).ToList());


                // Class accuracy - Dictionary
                var avgClassAccuracy = new Dictionary<string, float>();
                var minClassAccuracy = new Dictionary<string, float>();
                var maxClassAccuracy = new Dictionary<string, float>();
                foreach (var klass in workingDataset.Classes)
                {
                    foreach (var stats in stat)
                    {
                        if (avgClassAccuracy.ContainsKey(klass))
                        {
                            avgClassAccuracy[klass] += stats.ClassAccuracy[klass];
                            minClassAccuracy[klass] = Math.Min(minClassAccuracy[klass], stats.ClassAccuracy[klass]);
                            maxClassAccuracy[klass] = Math.Max(maxClassAccuracy[klass], stats.ClassAccuracy[klass]);
                        }
                        else
                        {
                            avgClassAccuracy.Add(klass, stats.ClassAccuracy[klass]);
                            minClassAccuracy.Add(klass, stats.ClassAccuracy[klass]);
                            maxClassAccuracy.Add(klass, stats.ClassAccuracy[klass]);
                        }
                    }
                    avgClassAccuracy[klass] = avgClassAccuracy[klass] / stat.Count;
                }


                float[,] avgConfusionArr = new float[workingDataset.ClassNum, workingDataset.ClassNum];
                float[,] minConfusionArr = new float[workingDataset.ClassNum, workingDataset.ClassNum];
                float[,] maxConfusionArr = new float[workingDataset.ClassNum, workingDataset.ClassNum];

                Array.Clear(avgConfusionArr, 0, avgConfusionArr.Length);
                minConfusionArr.Populate(float.MaxValue);
                maxConfusionArr.Populate(float.MinValue);

                foreach (var item in stat)
                {
                    for (int i = 0; i < workingDataset.ClassNum; i++)
                    {
                        for (int j = 0; j < workingDataset.ClassNum; j++)
                        {
                            avgConfusionArr[i, j] += item.Confusion[i][j] / stat.Count;
                            minConfusionArr[i, j] = Math.Min(minConfusionArr[i, j], item.Confusion[i][j]);
                            maxConfusionArr[i, j] = Math.Max(maxConfusionArr[i, j], item.Confusion[i][j]);
                        }
                    }
                }
                for (int i = 0; i < workingDataset.ClassNum; i++)
                {
                    for (int j = 0; j < workingDataset.ClassNum; j++)
                    {
                        avgConfusionArr[i, j] = (float)Math.Round(avgConfusionArr[i, j]);
                    }
                }
                var avgConfusion = ArrayToList(avgConfusionArr);
                var minConfusion = ArrayToList(minConfusionArr);
                var maxConfusion = ArrayToList(maxConfusionArr);


                var avgBestEpoch = bestEpoch.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgTotalTime = totalTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgTrainTime = trainTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgEvalTime = evalTime.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgAccuracy = accuracy.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgK1 = k1.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgK5 = k5.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgprecision = precision.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgfMeasure = fMeasure.Where(t => t.Key == model).Average(t => t.Value.Average());
                var avgrecall = recall.Where(t => t.Key == model).Average(t => t.Value.Average());


                var minBestEpoch = bestEpoch.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minTotalTime = totalTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minTrainTime = trainTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minEvalTime = evalTime.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minAccuracy = accuracy.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minK1 = k1.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minK5 = k5.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minprecision = precision.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minfMeasure = fMeasure.Where(t => t.Key == model).Min(t => t.Value.Min());
                var minrecall = recall.Where(t => t.Key == model).Min(t => t.Value.Min());

                var maxBestEpoch = bestEpoch.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxTotalTime = totalTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxTrainTime = trainTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxEvalTime = evalTime.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxAccuracy = accuracy.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxK1 = k1.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxK5 = k5.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxprecision = precision.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxfMeasure = fMeasure.Where(t => t.Key == model).Max(t => t.Value.Max());
                var maxrecall = recall.Where(t => t.Key == model).Max(t => t.Value.Max());


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
                    AvgClassAccuracy = avgClassAccuracy,
                    MaxClassAccuracy = maxClassAccuracy,
                    MinClassAccuracy = minClassAccuracy,
                    AvgConfusion = avgConfusion,
                    MaxConfusion = maxConfusion,
                    MinConfusion = minConfusion,
                    AvgFMeasure = avgfMeasure,
                    MaxFMeasure = maxfMeasure,
                    MinFMeasure = minfMeasure,
                    AvgPrecision = avgprecision,
                    MaxPrecision = maxprecision,
                    MinPrecision = minprecision,
                    AvgRecall = avgrecall,
                    MaxRecall = maxrecall,
                    MinRecall = minrecall,
                });
            }

            await SaveToJsonFile(averageResults, "averageResiscResults.json");

            //Console.WriteLine("Done!");
            //Console.ReadKey();

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

                    if (minArr.Count == j)
                    {
                        maxArr.Add(arrays[i][j]);
                        minArr.Add(arrays[i][j]);
                        avgArr.Add(arrays[i][j] / arrays.Where(a => a.Count > j).Count());
                    }
                    else
                    {
                        minArr[j] = Math.Min(arrays[i][j], minArr[j]);
                        maxArr[j] = Math.Max(arrays[i][j], maxArr[j]);
                        avgArr[j] += arrays[i][j] / arrays.Where(a => a.Count > j).Count();
                    }
                }
            }


            return (minArr, maxArr, avgArr);
        }

        public static List<List<T>> ArrayToList<T>(T[,] arr)
        {
            return arr.Cast<T>()
                      .Select((x, i) => new { x, index = i / arr.GetLength(1) })  // Use overloaded 'Select' and calculate row index.
                      .GroupBy(x => x.index)                                   // Group on Row index
                      .Select(x => x.Select(s => s.x).ToList())                  // Create List for each group.  
                      .ToList();
        }
    }

}
