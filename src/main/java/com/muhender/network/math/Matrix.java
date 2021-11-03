/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.muhender.network.math;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author R Muhender Raj
 */
public class Matrix {
    
    public double values[][];
    int x, y;
    
    /**
     * creates a x X y matrix and populates it with random values
     * @param x
     * @param y
     * @param randomize
     */
    public Matrix(int x, int y, boolean randomize){
        this(x, y);
        Random r = new Random();
        
        if(randomize){
            for(int i = 0; i < x; i++){
                for(int j = 0; j < y; j++){
                    values[i][j] = r.nextGaussian() / Math.sqrt(x);
     //                 values[i][j] = r.nextGaussian();
                }
            }
        }
        else{
            for(int i = 0; i < values.length; i++)
                Arrays.fill(values[i], 0);
        }
    }
    
    public Matrix(int x, int y, double arr[][]){
        this(x, y);
        for(int i = 0; i < x; i++){
            System.arraycopy(arr[i], 0, values[i], 0, y);
        }
    }
    
    public Matrix(int x, int y){
        this.x = x;
        this.y = y;
        values = new double[x][y];
    }
    
    public Matrix addToItself(Matrix m){
        if(m.x != x || m.y != y)
            throw new IllegalArgumentException("Matrices not addable");
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++){
                values[i][j] += m.values[i][j];
            }
        }
        
        return this;
    }
    
    public Matrix multiply(double d){
        Matrix m = new Matrix(x, y);
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++){
                m.values[i][j] = values[i][j] * d;
            }
        }
        
        return m;
    } 
    
    public Matrix multiply(Matrix other){
        if(this.y != other.x) throw new IllegalArgumentException("Not multipliable");
        Matrix m = new Matrix(this.x, other.y, false);
        
        for(int i = 0; i < m.x; i++){
            for(int j = 0; j < m.y; j++){
                for(int k = 0; k < this.y; k++){
                    m.values[i][j] += this.values[i][k] * other.values[k][j];
                } 
            }
        }
        
        return m;
    }
    
    public Matrix multiplyToItself(double d){
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++){
                values[i][j] *= d;
            }
        }
        
        return this;
    }    
    
    public Vector postMultiply(Vector v){
        if(v.dimension != y)
            throw new IllegalArgumentException("Not multipliable!");
        
        Vector vec = new Vector(x);
        
        for(int i = 0; i < x; i++){//something wrong here
            vec.values[i] = 0;
            for(int j = 0; j < y; j++){
                vec.values[i] += values[i][j] * v.values[j];
            }
        }
        return vec;
    }
    
    public Matrix transpose(){
        Matrix m = new Matrix(y, x);
        
        for(int i = 0; i < y; i++){
            for(int j = 0; j < x; j++){
                m.values[i][j] = values[j][i];
            }
        }
        
        return m;
    }
    
    public Matrix cofactor(int x, int y){    //in matrix form
        Matrix m = new Matrix(this.x - 1, this.y - 1);
        double d = ((x + y)%2 == 0)? 1: -1;
        
        for(int i = 0, a = 0; a < this.x - 1; i++, a++){
            for(int j = 0, b = 0; b < this.y - 1; j++, b++){
                if(i == x) i++;
                if(j == y) j++;
                
                m.values[a][b] = d * this.values[i][j];
                
            }
        
        }
        
        return m;
    }
    
    public static double determinant(Matrix m){
        if(m.x != m.y) throw new IllegalArgumentException("Not a square matrix");
        if(m.x == 1)
            return m.values[0][0];
        
        double det = 0;
        
        for(int j = 0; j < m.y; j++){
            det += m.values[0][j] * determinant(m.cofactor(0, j));
        }        
        
        return det;
    }
    
    public Matrix adjugate(){
        if(this.x != this.y) throw new IllegalArgumentException("Not a square matrix");
        Matrix m = new Matrix(this.y, this.x);
        
        for(int i = 0; i < this.x; i++){
            for(int j = 0; j < this.y; j++){
                m.values[i][j] = Matrix.determinant(this.cofactor(j, i));
            }
        }
        
        return m;
    }
    
    public Matrix inverse(){
        return this.adjugate().multiplyToItself(1 / Matrix.determinant(this));
    }
    
    public Matrix cloneItself(){
        Matrix m = new Matrix(x, y);
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++){
                m.values[i][j] = values[i][j];
            }
        }
        return m;
    }
    
    @Override
    public String toString(){
        String s = "";
        for(int i = 0; i < this.x; i++){
            for(int j = 0; j < this.y; j++)
                s += this.values[i][j] + "\t";
            s += "\n";
        }
        
        return s;
    }
    
    public Matrix polyakAverageToItself(double tau, Matrix other){
        this.multiplyToItself(1 - tau).addToItself(other.multiply(tau));
        return this;
    }

}
