/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package oopp;

/**
 *
 * @author admin
 */
class NeuronNetwork{
    
    double[][] pattern_ = {{4,10},{10,6},{2,1},{-4,4}};
    double[] answer_ = {1,0,0,1};
    double[] answer2_ = {1,1,0,0};
    
    OOPP perceptron1;
    OOPP perceptron2;
    
    NeuronNetwork(){
        perceptron1 = new OOPP(TypeLine.FIRST, pattern_, answer_);
        perceptron2 = new OOPP(TypeLine.SECOND, pattern_, answer2_);
    }
    
    public String checkPoint(double[] point){
        perceptron1.point(point);
        perceptron2.point(point);      
 
        return Integer.toString(perceptron1.getOutputData())+Integer.toString(perceptron2.getOutputData());
    }
    
}

class ActivationFunctions{
    public double functionFirstLine(double x){
        double returnValue = 0;
        
        if(x > 12/Math.sqrt(13)){
            returnValue = 1;
        } else {
            returnValue = 0;
        }
        
        return returnValue;
    }
    
    public double functionSecondLine(double x){
        double returnValue = 0;
        
        if(x > 72/Math.sqrt(145)){
            returnValue = 1;
        } else {
            returnValue = 0;
        }
        
        return returnValue;
    } 
}

enum TypeLine{FIRST,SECOND};
enum TypeLayer{INPUT, HIDDEN, OUTPUT};

class Layer{
    public int countNeuron_;
    public double[] neurons_;
    public double[] errors_;
    public double[][] weights_;
    
    Layer(int countNeurons){
        countNeuron_ = countNeurons;
        neurons_ = new double[countNeurons];
        errors_ = new double[countNeurons];
    }
    
    public void initWeights(int countNeuronOfNeighboringLayer){
        weights_ = new double[countNeuron_][countNeuronOfNeighboringLayer];
        for(int i = 0; i < weights_.length; i++){
            for(int j = 0; j < weights_[i].length; j++){
                weights_[i][j] = Math.random()*0.1+0.01;
            }
        }
    }
}

public class OOPP {
    
    double[][] test = {{-2,-2},{-4,4}};
    public TypeLine typeLine_;
       
    double[][] pattern_;
    double[] answer_;
    
    double speedStudy_ = 0.1;
    
    Layer[] layers_;
   
    
    OOPP(TypeLine typeLine, double[][] pattern, double[] answer){
        pattern_ = java.util.Arrays.copyOf(pattern, pattern.length);
        answer_ = java.util.Arrays.copyOf(answer, answer.length);
        
        typeLine_ = typeLine;
        createLayers(2);
        createConnectionBetweenLayers();
  
        study();
    }
    
    public void createLayers(int countLayers){
        layers_ = new Layer[countLayers];
        
        
        layers_[0] = new Layer(2);//count neurons on first layer = 2
        layers_[1] = new Layer(1);//count neurons on second layer = 1
    }
    
    public void createConnectionBetweenLayers(){
        for(int lay = 0; lay < layers_.length-1; lay++){
            layers_[lay].initWeights(layers_[lay+1].countNeuron_);
        }
    }
    
    public void sendSignals(double[] signals){
        for(int i = 0; i < layers_[0].countNeuron_; i++){
            layers_[TypeLayer.INPUT.ordinal()].neurons_[i] = signals[i];
        }
    }
    
    public int getOutputData(){
        return (int)layers_[layers_.length - 1].neurons_[layers_[layers_.length - 1].neurons_.length - 1];
    }
    
    public void calc(){
        for(int numLayer = 1; numLayer < layers_.length; numLayer++){
            for(int numNeuron = 0; numNeuron < layers_[numLayer].countNeuron_; numNeuron++){
                layers_[numLayer].neurons_[numNeuron] = 0;
                for(int numBeforeNeuron = 0; numBeforeNeuron < layers_[numLayer-1].countNeuron_; numBeforeNeuron++){
                    layers_[numLayer].neurons_[numNeuron] += layers_[numLayer-1].neurons_[numBeforeNeuron]*layers_[numLayer-1].weights_[numBeforeNeuron][numNeuron];
                }
                
                //activation function:
                if(typeLine_.ordinal() == TypeLine.FIRST.ordinal()){
                    layers_[numLayer].neurons_[numNeuron] = new ActivationFunctions().functionFirstLine(layers_[numLayer].neurons_[numNeuron]);
                } else {
                    layers_[numLayer].neurons_[numNeuron] = new ActivationFunctions().functionSecondLine(layers_[numLayer].neurons_[numNeuron]);
                }
            } 
        }
    }
     
    public void point(double[] point){
        sendSignals(point);
        calc();
    }
    
    public void study(){
        double gE = 0;
        do{
            gE = 0;
            for(int numP = 0; numP < pattern_.length; numP++){
                sendSignals(pattern_[numP]);
                calc();
                
                double lError = answer_[numP] - layers_[layers_.length - 1].neurons_[layers_[layers_.length - 1].countNeuron_-1];
                gE += Math.abs(lError);
                
                //local resulted pattern error:
                for(int i = 0; i < layers_[layers_.length - 1].errors_.length; i++){
                    layers_[layers_.length - 1].errors_[i] = answer_[numP] - layers_[layers_.length - 1].neurons_[layers_[layers_.length - 1].countNeuron_ - 1];
                }
                
                //calculate errors value on every layer:
                for(int numLayer = layers_.length - 2; numLayer > TypeLayer.INPUT.ordinal(); numLayer--){
                    for(int i = 0; i < layers_[numLayer].errors_.length; i++){
                        layers_[numLayer].errors_[i] = 0;
                        for(int j = 0; j < layers_[numLayer+1].errors_.length; j++){
                            for(int numW = 0; numW < layers_[numLayer].weights_[i].length; numW++){
                                layers_[numLayer].errors_[i] += layers_[numLayer+1].errors_[j]*layers_[numLayer].weights_[i][numW];
                            }
                        }
                    }
                }
                
                //corrective:
                for(int numLayer = 0; numLayer < layers_.length - 1; numLayer++){
                    for(int numW = 0; numW < layers_[numLayer].weights_.length; numW++){
                        for(int i = 0; i < layers_[numLayer].weights_[numW].length; i++){
                            layers_[numLayer].weights_[numW][i] += speedStudy_*layers_[numLayer+1].errors_[i]*layers_[numLayer].neurons_[numW];                        }
                    }
                }                    
            }
            //System.out.println("error: "+gE);
        }while(gE > 0);
    }
    
    public void test(){
        for(int p = 0; p < test.length; p++){
            sendSignals(test[p]);
            calc();

            System.out.println("outer: "+layers_[layers_.length-1].neurons_[layers_[layers_.length-1].countNeuron_-1]);       
        }
    }

    /**
     * @param args the command line arguments
     */
        
    public static void main(String[] args) {
    int BINARY = 2; 
    double[] point = {0,12};    
    
    NeuronNetwork nn = new NeuronNetwork();
    
    System.out.println("Object(point): (" +point[0]+"; "+point[1]+") is exsist class: "+(Integer.parseInt(nn.checkPoint(point),BINARY)+1));
    
    }
    
}
