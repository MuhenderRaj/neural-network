/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.muhender.network.math;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 *
 * @author R Muhender Raj
 */
public class Vector {
    
    int dimension;
    public double values[];
    
    public Vector(int dim, double args[]){
        this(dim);
        
        for(int i = 0; i < dim; i++){
            values[i] = args[i];
        }
    }
    
    public Vector(int dim, boolean randomize){
        this(dim);
        Random r = new Random();
        
        if(randomize){
            for(int i = 0; i < dim; i++){
                values[i] = r.nextGaussian();
            }
        }
        else{
            Arrays.fill(values, 0);
        }
    }
    
    public Vector(int dim){
        dimension = dim;
        values = new double[dim];
    }
    
    public Vector add(Vector v){
        if(v.dimension != dimension)
            throw new IllegalArgumentException("Not addable!");
        
        Vector vec = new Vector(dimension);
        
        
        for(int i = 0; i < dimension; i++){
            vec.values[i] = v.values[i] + values[i];
        }
        
        return vec;
    }
    
    public Vector addToItself(Vector v){
        if(v.dimension != dimension)
            throw new IllegalArgumentException("Not addable!");
        
        for(int i = 0; i < dimension; i++){
            values[i] += v.values[i];
        }
        
        return this;
    }
    
    public Vector multiply(double d){
        Vector vec = new Vector(dimension);
        
        for(int i = 0; i < dimension; i++){
            vec.values[i] = values[i] * d;
        }
        
        return vec;
    }
    
    public Vector multiplyToItself(double d){
        for(int i = 0; i < dimension; i++){
            values[i] *= d;
        }
        
        return this;
    }
    
    public double magnitudeSquared(){
        double sum = 0;
        
        for(int i = 0; i < dimension; i++){
            sum += values[i] * values[i];
        }
        
        return sum;
    }
    
    public Vector unitVector(){
        double magnitude = Math.sqrt(this.magnitudeSquared());
        
        for(int i = 0; i < dimension; i++){
            values[i] /= magnitude;
        }
        
        return this;
    }
    
    public double sum(){
        double sum = 0;
        
        for(int i = 0; i < dimension; i++){
            sum += values[i];
        }
        
        return sum;
    }
    
    public Vector hadamardProduct(Vector v){
        if(v.dimension != dimension)
            throw new IllegalArgumentException("Dimensions don't match!");
        
        Vector vec = new Vector(dimension);
        
        for(int i = 0; i < dimension; i++){
            vec.values[i] = v.values[i] * values[i];
        }
        
        return vec;
    }
    
    public Matrix outerProduct(Vector v){
        int x = dimension;
        int y = v.dimension;
        
        Matrix m = new Matrix(x, y);
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++){
                m.values[i][j] = values[i] * v.values[j];
            }
        }
        
        return m;
    }
    
    /**
     *
     * @param function the function to apply to each element of the vector. <code>(x) -> 1 / (1 + Math.exp(-x))</code> by default
     * @return
     */
    public Vector applyFunction(Function<Double, Double> function){
        if(function == null)
            function = (x) -> 1 / (1 + Math.exp(-x));    //default is sigmoid logistic
        
        Vector vec = new Vector(dimension);
        
        for(int i = 0; i < dimension; i++){
            vec.values[i] = function.apply(values[i]);
        }
        
        return vec;
    }
    
    @Override
    public String toString(){
        return Arrays.toString(values);
    }
    
    public Vector cloneItself(){
        Vector v = new Vector(dimension);
        
        for(int i = 0; i < dimension; i++){
            v.values[i] = values[i];
        }
        return v;
    }
    
    public Vector polyakAverageToItself(double tau, Vector other){
        this.multiplyToItself(1 - tau).addToItself(other.multiply(tau));
        return this;
    }
}
