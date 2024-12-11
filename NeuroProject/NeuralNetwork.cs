using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Messaging;
using System.Runtime.Serialization.Formatters;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeuroProject
{
    public class NeuralNetwork
    {
        public Topoligy Topoligy { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topoligy topoligy)
        {
            Topoligy = topoligy;
            
            Layers = new List<Layer>();
            // Создание слоёв
            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutoutLayer();
        }

        //принимает входящие сигналы и пушит их / прогоняет сигналы
        public Neuron FeedForward(params double[] inputSignals)
        {
            //этот метод отсылает сигнал в нейрон
            SendSignalsToInputNeurons(inputSignals);

            //пробегается по всем слоям после пуша значений внутрь
            FeedForwardAllLayersAfterInput();

            // ну тут по хуйне. Если выходное значение Topology = 1, то возвращаем 0й нейрон на последний слой
            if (Topoligy.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataset">одна эпоха обучения</param>
        /// <param name="epoch">колличество эпох / сколько раз будем прогонять коллекцию</param>
        /// <returns></returns>
        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                foreach(var data in dataset)
                {
                    error += Backpropagation(data.Item1, data.Item2);
                }
            }
            //возвращаем среднюю ошибку
            var res = error / epoch;
            return res;
        }

        //метод обратного распространения ошибки
        /// <summary>
        /// 
        /// </summary>
        /// <param name="expected"> ожидаемое значение</param>
        /// <param name="inputs">набор входных параметров/входные сигналы</param>
        /// <returns></returns>
        private double Backpropagation(double expected, params double[] inputs)
        {
            //отправили входные данные и получили какой-то результат
            var actual = FeedForward(inputs).Output;
            // вычисляем раздницу между актуальным и ожидаемым значением
            var differents = actual - expected;
            foreach (var neuron in Layers.Last().Neurons)
            {
                //обучаем нейрон. Передаем разницу в differents
                neuron.Learn(differents, Topoligy.LearningRate); 
            }

            //счетчик идет на уменьшение т.к считаем с права на лево
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];

                //берем предыдущий слой
                var previousLayer = Layers[j - 1];
                for (int i = 0; i < layer.NeuronCount; i++) // просматриваем все нейроны на слое
                {
                    //из текущего слоя берем нейрон в рамках слоя
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        //берем i для weight потому что нужен нейрон предыдущего слоя
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        //на значении ошибки обучаем ТЕКУЩИЙ нейрон
                        neuron.Learn(error, Topoligy.LearningRate);
                    }
                }
            }
            var res = differents * differents;
            return res;
            
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }


        // Создаем выходой слой (OutputLayer)
        private void CreateOutoutLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topoligy.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topoligy.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topoligy.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron); 
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
            
        }

        
        public void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topoligy.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
    }
}
