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
        public static Dataset Resisc => new Dataset("RESISC45", new List<string> { "airplane",
"airport",
"baseball_diamond",
"basketball_court",
"beach",
"bridge",
"chaparral",
"church",
"circular_farmland",
"cloud",
"commercial_area",
"dense_residential",
"desert",
"forest",
"freeway",
"golf_course",
"ground_track_field",
"harbor",
"industrial_area",
"intersection",
"island",
"lake",
"meadow",
"medium_residential",
"mobile_home_park",
"mountain",
"overpass",
"palace",
"parking_lot",
"railway",
"railway_station",
"rectangular_farmland",
"river",
"roundabout",
"runway",
"sea_ice",
"ship",
"snowberg",
"sparse_residential",
"stadium",
"storage_tank",
"tennis_court",
"terrace",
"thermal_power_station",
"wetland",});
        public Dataset(string name, List<string> classes)
        {
            Name = name;
            Classes = classes;
        }


    }
}
