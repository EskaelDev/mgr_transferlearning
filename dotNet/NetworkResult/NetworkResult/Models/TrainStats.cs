using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NetworkResult.Models
{
    class TrainStats
    {
        [JsonPropertyName("accuracy")]
        public float Accuracy { get; set; }
        
        [JsonPropertyName("k1")]
        public float K1 { get; set; }
        
        [JsonPropertyName("k5")]
        public float K5 { get; set; }
        
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; }
        
        [JsonPropertyName("train_loss_array")]
        public List<float> TrainLoss { get; set; }
        
        [JsonPropertyName("valid_loss_array")]
        public List<float> ValidLoss{ get; set; }
        
        [JsonPropertyName("train_accuracy_array")]
        public List<float> TrainAccuracy { get; set; }
        
        [JsonPropertyName("valid_accuracy_array")]
        public List<float> ValidAccuracy{ get; set; }
        
        [JsonPropertyName("best_epoch")]
        public int BestEpoch { get; set; }
        
        [JsonPropertyName("total_time")]
        public float TotalTime{ get; set; }
        
        [JsonPropertyName("train_time_sum")]
        public float TrainTime{ get; set; }
        
        [JsonPropertyName("eval_time_sum")]
        public float EvalTime{ get; set; }
    }
}
