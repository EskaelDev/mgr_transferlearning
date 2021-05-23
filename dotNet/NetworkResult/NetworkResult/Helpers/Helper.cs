using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NetworkResult.Helpers
{
    public static class Helper
    {
        public static List<string> Models => new List<string> {"cifar_densenet121",
                                                               "cifar_densenet161",
                                                               "cifar_densenet169",
                                                               "cifar_googlenet",
                                                               "cifar_mobilenet_v2",
                                                               "cifar_resnet18",
                                                               "cifar_resnet34",
                                                               "cifar_resnet50",
                                                               "cifar_vgg11_bn",
                                                               "cifar_vgg13_bn",
                                                               "cifar_vgg16_bn",
                                                               "cifar_vgg19_bn",
                                                               "densenet121",
                                                               "densenet161",
                                                               "densenet169",
                                                               "googlenet",
                                                               "inception_v3",
                                                               "mobilenet_v2",
                                                               "resnet18",
                                                               "resnet34",
                                                               "resnet50",
                                                               "vgg11_bn",
                                                               "vgg13_bn",
                                                               "vgg16_bn",
                                                               "vgg19_bn"};
    }
}
