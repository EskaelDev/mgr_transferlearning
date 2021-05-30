using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NetworkResult.Models
{
    class AverageResult
    {
        public string ModelName { get; set; }


        public int AvgBestEpoch { get; set; }
        public float AvgTotalTime { get; set; }
        public float AvgTrainTime { get; set; }
        public float AvgEvalTime { get; set; }
        public float AvgAccuracy { get; set; }
        public float AvgK1 { get; set; }
        public float AvgK5 { get; set; }


        public int MaxBestEpoch { get; set; }
        public float MaxTotalTime { get; set; }
        public float MaxTrainTime { get; set; }
        public float MaxEvalTime { get; set; }
        public float MaxAccuracy { get; set; }
        public float MaxK1 { get; set; }
        public float MaxK5 { get; set; }

        public int MinBestEpoch { get; set; }
        public float MinTotalTime { get; set; }
        public float MinTrainTime { get; set; }
        public float MinEvalTime { get; set; }
        public float MinAccuracy { get; set; }
        public float MinK1 { get; set; }
        public float MinK5 { get; set; }

        public List<float> MinTrainLoss { get; set; }
        public List<float> MinValidLoss { get; set; }
        public List<float> MinTrainAccuracy { get; set; }
        public List<float> MinValidAccuracy { get; set; }

        public List<float> MaxTrainLoss { get; set; }
        public List<float> MaxValidLoss { get; set; }
        public List<float> MaxTrainAccuracy { get; set; }
        public List<float> MaxValidAccuracy { get; set; }

        public List<float> AvgTrainLoss { get; set; }
        public List<float> AvgValidLoss { get; set; }
        public List<float> AvgTrainAccuracy { get; set; }
        public List<float> AvgValidAccuracy { get; set; }



        public Dictionary<string, float> AvgClassAccuracy { get; set; }
        public Dictionary<string, float> MaxClassAccuracy { get; set; }
        public Dictionary<string, float> MinClassAccuracy { get; set; }

        public List<List<float>> AvgConfusion { get; set; }
        public List<List<float>> MaxConfusion { get; set; }
        public List<List<float>> MinConfusion { get; set; }

        public float AvgFMeasure { get; set; }
        public float MaxFMeasure { get; set; }
        public float MinFMeasure { get; set; }

        public float AvgPrecision { get; set; }
        public float MaxPrecision { get; set; }
        public float MinPrecision { get; set; }

        public float AvgRecall { get; set; }
        public float MaxRecall { get; set; }
        public float MinRecall { get; set; }
    }
}
