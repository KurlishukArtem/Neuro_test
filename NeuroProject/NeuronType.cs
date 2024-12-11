using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroProject
{
    public enum NeuronType
    {
        Input = 0, //входящие значения
        Normal = 1, // значения процесса (скрытый слой)
        Output = 2 // выходные данные
    }
}
