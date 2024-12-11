using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting;
using System.Runtime.Serialization.Formatters;
using System.Text;
using System.Threading.Tasks;

namespace NeuroProject
{
    public class Neuron
    {
        //тут объявляем переменные
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightRandomValue(inputCount);
        }

        private void InitWeightRandomValue(int inputCount)
        {
            var rnd = new Random();
            //здесь закидываем значения
            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                    Weights.Add(rnd.NextDouble());
                Inputs.Add(0);

            }
        }

        //сюда попадают все input'ы
        public double FeedForward(List<double> inputs)
        {
            //прогоняеем все инпуты и ничего с ними не делаем
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum+= inputs[i] * Weights[i];

            }
            //если тип нейрона != входящему типу, то считаем сигмойду в ином случае - выводим сумму
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }
        // Метод с формулой сигмойды
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        //производная от функции сигмойды
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        

        //выполнение изменения коэфициентов/изменение нейрона
        public void Learn(double error, double learningRate)
        {
            // если тип входных данных совпадает с NeuronType => возвращаем значение
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            // формула дельты
            var delta = error * SigmoidDx(Output);

            // прогоняет значения до условного значения "Веса" по i - как единица общего списка
            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * delta * learningRate;
                Weights[i] = newWeight;
            }

            Delta = delta;
        }

        //перезаписываем выходные данные
        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
