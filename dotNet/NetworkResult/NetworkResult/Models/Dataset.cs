using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NetworkResult.Models
{
    class Dataset
    {

        public string Name { get; set; }
        public List<string> Classes { get; set; }
        public int ClassNum => Classes.Count;

        public static Dataset UcLanduse => new Dataset("UCMerced_LandUse", new List<string> { "airplane", "chaparral", "forest", "buildings", "agricultural", "baseballdiamond", "denseresidential", "beach", "freeway", "parkinglot", "harbor", "golfcourse", "intersection", "overpass", "mobilehomepark", "mediumresidential", "river", "runway", "storagetanks", "sparseresidential", "tenniscourt" });
        public Dataset(string name, List<string> classes)
        {
            Name = name;
            Classes = classes;
        }


    }
}
