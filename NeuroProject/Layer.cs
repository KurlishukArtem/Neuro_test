﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeuroProject
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;

        public NeuronType Type;

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal) 
        {
            // проверить все входные нейроны на соответствие типу

            Neurons = neurons;
            Type = type;
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
