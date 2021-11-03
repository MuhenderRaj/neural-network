/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.muhender.network;

import com.muhender.network.math.Matrix;
import com.muhender.network.math.Vector;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 *
 * @author R Muhender Raj
 */
public class NeuralNetwork {
    
    public Matrix weights[];    //don't use weights[0]
    public Vector biases[];     //don't use biases[0]
    Vector zInput[];     //don't use zInput[0] or zInput[1]
    Vector activations[];//don't use activations[0]
    public int layers;
    public int nodesNum[];
    public static Function<Double, Double> activationFunc, activationDerivative, sigmoid, sigmoidDerivative, ReLU, ReLUDerivative;
    public int func;
    BiFunction<Vector, Vector, Double> cost;
    BiFunction<Vector, Vector, Vector> costDerivative;
    Vector errors[];    //don't use errors[0]
    Matrix weightGradient[];
    Vector biasGradient[];
    Vector requiredOutput;
    
    double learningRate;
    
    public NeuralNetwork(int layers, int nodesNum[], double learningRate, int activation){
        this(layers, nodesNum, new double[nodesNum[0]], new double[nodesNum[layers - 1]], learningRate, activation);
    }
    
    public NeuralNetwork(int layers, int nodesNum[], double inputs[], double output[], double learningRate, int activation){
        weights = new Matrix[layers];
        biases = new Vector[layers];
        zInput = new Vector[layers];
        activations = new Vector[layers];
        activations[0] = new Vector(nodesNum[0], inputs);
        errors = new Vector[layers];
        
        this.layers = layers;
        this.nodesNum = new int[layers];
        
        requiredOutput = new Vector(nodesNum[layers - 1]);
        
        weightGradient = new Matrix[layers];
        biasGradient = new Vector[layers];
        
        this.learningRate = learningRate;
        
        //neuron activation functions
        sigmoid = x -> 1 / (1 + Math.exp(-x));
        sigmoidDerivative = x -> sigmoid.apply(x) * (1 - sigmoid.apply(x));
        ReLU = x -> (x > 0)? x: 0.0;   //see
        ReLUDerivative = x -> (x > 0) ? 1.0: 0.0;
        
        switch(activation){
            case 1:
                activationFunc = sigmoid;
                activationDerivative = sigmoidDerivative;
                break;
                
            case 2:
                activationFunc = ReLU;
                activationDerivative = ReLUDerivative;
                break;
        }
        
        func = activation;
        
        
        //quadratic cost
        cost = (y, z) -> (y.add(z.multiply(-1))).magnitudeSquared() / 2;    //y is the required last layer zInput
        costDerivative = (y, z) -> z.add(y.multiply(-1));                   //z is the actual zInput
        
        modifyExpectations(output);
        
        this.nodesNum[0] = nodesNum[0];
        
        for(int i = 1; i < layers; i++){
            weights[i] = new Matrix(nodesNum[i - 1], nodesNum[i], true);    //weight is wi[x][y]. This connects xth neuron in layer i - 1 to yth neuron in layer i
            biases[i] = new Vector(nodesNum[i], false);
            this.nodesNum[i] = nodesNum[i];
        }
        
        //feedForward();
    }
    
    public NeuralNetwork feedForward(){
        for(int i = 1; i < layers; i++){
            zInput[i] = weights[i].transpose()
                    .postMultiply(activations[i - 1])
                    .add(biases[i]);
            activations[i] = zInput[i].applyFunction(activationFunc);
        }
        
        return this;
    }

    public NeuralNetwork gradientDescent(){
        for(int i = 1; i < layers; i++){
            biases[i].addToItself(biasGradient[i].multiplyToItself(-learningRate));
            
            weights[i].addToItself(weightGradient[i].multiplyToItself(-learningRate));
        }
        
        return this;
    }
    
    public NeuralNetwork backPropagate(){
        //BP1
        biasGradient[layers - 1] = errors[layers - 1] 
                = costDerivative.apply(requiredOutput, zInput[layers - 1])
                        .hadamardProduct(zInput[layers - 1].applyFunction(activationDerivative));
        
        
        //BP4
        weightGradient[layers - 1] = activations[layers - 2].outerProduct(errors[layers - 1]);
        
        for(int i = layers - 2; i > 0; i--){
            //BP2
            errors[i] = weights[i + 1].postMultiply(errors[i + 1])
                    .hadamardProduct(zInput[i].applyFunction(activationDerivative));
            
            //BP3
            biasGradient[i] = errors[i];   
            
            //BP4
            weightGradient[i] = activations[i - 1].outerProduct(errors[i]);
        }
        
        return this;
    }
    
    public double calculateCost(){
        return cost.apply(requiredOutput, zInput[zInput.length - 1]);
    }
       
    public Vector getOutputs(){
        return zInput[zInput.length - 1];
    }
    
    public NeuralNetwork modifyInputs(double arr[]){
        System.arraycopy(arr, 0, activations[0].values, 0, nodesNum[0]);
        return this;
    }
    
    public NeuralNetwork modifyExpectations(double arr[]){
        requiredOutput.values = arr.clone();
        return this;
    }
    
    public NeuralNetwork cloneNetwork(){
        NeuralNetwork n = new NeuralNetwork(layers, nodesNum, activations[0].values, requiredOutput.values, learningRate, func);
        for(int i = 1; i < layers; i++){
            n.weights[i] = weights[i].cloneItself();
            n.biases[i] = biases[i].cloneItself();
        }
        
        return n;
    }
    
    public NeuralNetwork polyakAverageToItself(double tau, NeuralNetwork other){
        for(int i = 1; i < layers; i++){
            weights[i].polyakAverageToItself(tau, other.weights[i]);
            biases[i].polyakAverageToItself(tau, other.biases[i]);
        }
        
        return this;
    }
}
