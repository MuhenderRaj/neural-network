/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.muhender.network.agents;

import com.muhender.network.NeuralNetwork;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/**
 * The genetic algorithm. Get the champion in a batch, mutate, repeat
 * @author R Muhender Raj
 */
public class Genetic {
    public int popSize;
    NeuralNetwork brains[];
    public double scores[];
    double parameters[][];
    int champion;
    int generation;
    double mutationRate;
    Map<Integer, String> outputsToActions;
    public boolean alive[];
    Random r;
    double maxMutation;
    
    public Genetic(int popSize, double mutationRate, int nodesNum[], Map<Integer, String> outputsToActions){
        this.popSize = popSize;
        brains = new NeuralNetwork[popSize];
        scores = new double[popSize];
        alive = new boolean[popSize];
        parameters = new double[popSize][nodesNum[0]];
        generation = 1;
        this.mutationRate = mutationRate;
        this.outputsToActions = outputsToActions;
        r = new Random();
        maxMutation = 50;
        
        for(int i = 0; i < popSize; i++){
            scores[i] = 0;
            brains[i] = new NeuralNetwork(nodesNum.length, nodesNum, 0, 1);
            alive[i] = true;
            
        }
    }
    
    public int getChampion(){
        champion = 0;
        
        for(int i = 0; i < popSize; i++)
            if(scores[i] > scores[champion])
                champion = i;
        
        return champion;
    }
    
    public NeuralNetwork getChampionNetwork(){
        return brains[getChampion()];
    }
    
    public double getChampionScore(){
        return scores[getChampion()];
    }
    
    public void setParameters(double[][] inputs){
        for(int i = 0; i < popSize; i++){
            System.arraycopy(inputs[i], 0, parameters[i], 0, brains[0].nodesNum[0]);
        }
    }
    
    public String[] getActions(){        
        double temp[];
        String actions[] = new String[popSize];
        
        for(int i = 0; i < popSize; i++){
            if(alive[i]){
                temp = brains[i].modifyInputs(parameters[i])
                        .feedForward()
                        .getOutputs().values;

                int maxIndex = 0;
                
                for(int j = 0; j < temp.length; j++){
                    if(temp[maxIndex] < temp[j]){
                        temp[maxIndex] = temp[j];
                        maxIndex = j;
                    }
                }
                
                actions[i] = outputsToActions.get(maxIndex);
            }
            else{
                actions[i] = "";
            }
        }
        
        return actions;
    }
    
    public void destroyIndividual(int i){
        if(alive[i])
            alive[i] = false;
        else
            System.out.println("Already destroyed");
    }
    
    public boolean generationCompleted(){
        for(boolean b : alive)
            if(b == true)
                return false;
        
        return true;
    }
    
    public void mutateChampion(){
        int champ = getChampion();
        
        for(int i = 0; i < popSize; i++){
            if(i == champ)
                continue;
            
            for(int j = 1; j < brains[champion].weights.length; j++){
                brains[i].weights[j] = brains[champion].weights[j].cloneItself();
                brains[i].biases[j] = brains[champion].biases[j].cloneItself();
                
                for(int k = 0; k < brains[champion].weights[j].values.length; k++){    
                    for(int l = 0; l < brains[champion].weights[j].values[0].length; l++){    
                        if(r.nextDouble() < mutationRate){
                            brains[i].weights[j].values[k][l] = r.nextDouble() * maxMutation * ((r.nextBoolean())? -1 : 1);
                        }
                    }
                }
            }
        }
    }
    
    public void updateScores(int i, double state[], int goal[]){
        double distSq = 0;
        
        for(int k = 0; k < state.length; k++){
            distSq += Math.pow(state[k] - goal[k], 2);
        }
        
        scores[i] = 100 / distSq;
    }
    
    public void nextGen(){
        
        for(int i = 0; i < popSize; i++){
            alive[i] = true;
            scores[i] = 0;
        }
    }
    
}
