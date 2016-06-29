package com.dap.dl4j;

import java.util.Random;
import java.util.Scanner;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.dap.routines.basic.SystemRoutines;

public class TestRL {

	DeepQNetwork RLNet;
	int size = 4;
	int scale = 3;
	
	float FrameBuffer[][];
	
	void InitNet(){
		
		int InputLength = size*size*2+1 ;
		int HiddenLayerCount = 150 ;
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
       		 .seed(123)
	             .iterations(1)
	             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	             .learningRate(0.0025)
	             .updater(Updater.NESTEROVS).momentum(0.95)
	             .list()
	             .layer(0, new DenseLayer.Builder().nIn(InputLength).nOut(HiddenLayerCount)
	            		 	.weightInit(WeightInit.XAVIER)
		                        .activation("relu")
		                        .build())
	             .layer(1, new OutputLayer.Builder(LossFunction.MSE)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("identity").weightInit(WeightInit.XAVIER)
	                        .nIn(HiddenLayerCount).nOut(4).build())
	             .pretrain(false).backprop(true).build();
		
		
		RLNet = new DeepQNetwork(conf1 ,  100000 , .99f , 1d , 1024 , 500 , 1024 , InputLength , 4);
	}
	
	Random r = new Random();
	
	float[][] GenerateMap(){
		int player = r.nextInt(size * size);
		int goal = r.nextInt(size * size);
		while(goal == player)
			goal = r.nextInt(size * size);
		float[][] map = new float[size][size];
		for(int i = 0; i < size*size ; i++)
			map[i/size][i%size] = 0;
		map[goal/size][goal%size] = -1 ;
		map[player/size][player%size] = 1 ;

		
		return map;
	}
	
	int CalcPlayerPos(float[][] Map){
		int x = -1;
		for(int i = 0 ; i < size * size ; i++){
			if(Map[i/size][i%size] == 1)
				return i;
		}
		return x;
	}
	
	int CalcGoalPos(float[][] Map){
		int x = -1;
		for(int i = 0 ; i < size * size ; i++){
			if(Map[i/size][i%size] == -1)
				return i;
		}
		return x;
	}
	
	int[] GetActionMask(float[][] CurrMap){
		int retVal[] = { 1 , 1 , 1 , 1} ;
		
		int player = CalcPlayerPos(CurrMap);
		if(player < size)
			retVal[0] = 0;
		if(player >= size*size - size )
			retVal[1] = 0;
		if(player % size == 0)
			retVal[2] = 0;
		if(player % size == size-1)
			retVal[3] = 0;
		
		return retVal ;
	}
	
	float[][] DoMove(float[][] CurrMap , int action){
		float nextMap[][] = new float[size][size];
		for(int i = 0; i < size*size ; i++)
			nextMap[i/size][i%size] = CurrMap[i/size][i%size];
		
		int player = CalcPlayerPos(CurrMap);
		nextMap[player/size][player%size] = 0;
		if( action == 0 ){
			if(player - size >= 0)
				nextMap[(player-size)/size][player%size] = 1;
			else
				SystemRoutines.Exit("Bad Move");
		}
		else if( action == 1 ){
			if(player + size < size * size)
				nextMap[(player+size)/size][player%size] = 1;
			else
				SystemRoutines.Exit("Bad Move");
		}
		else if( action == 2 ){
			if((player%size) - 1 >= 0)
				nextMap[player/size][(player%size) - 1] = 1;
			else
				SystemRoutines.Exit("Bad Move");
		}
		else if( action == 3 ){
			if((player%size) + 1 < size)
				nextMap[player/size][(player%size) + 1] = 1;
			else
				SystemRoutines.Exit("Bad Move");
		}
		return nextMap;
	}
	
	float CalcReward(float[][]CurrMap , float[][]NextMap){
		int newGoal = CalcGoalPos(NextMap);
		
		if(newGoal == -1 )
			return size*size + 1;
		
		return -1f;
	}
	
	void AddToBuffer(float[][] NextFrame){
			FrameBuffer = NextFrame;
	}
	
	INDArray FlattenInput(int TimeStep){
		float flattenedInput[] = new float[size*size*2+1];
		for(int a = 0; a < size ; a++){
			for(int b = 0; b < size ; b++){
				if(FrameBuffer[a][b] == -1)
					flattenedInput[a*size + b] = 1;
				else
					flattenedInput[a*size + b] = 0;
				if(FrameBuffer[a][b] == 1)
					flattenedInput[size*size + a*size + b] = 1;
				else
					flattenedInput[size*size + a*size + b] = 0;
			}
		}
		flattenedInput[size*size*2] = TimeStep;
		return Nd4j.create(flattenedInput);
	}
	
	void PrintBoard(float[][] Map){
		for(int x = 0; x < size ; x++){
			for(int y = 0; y < size ; y++){
				System.out.print((int)Map[x][y]);
			}
			System.out.println("");
		}
		System.out.println("");
	}
	
	public static void main(String[] args) {
		
		TestRL test = new TestRL() ;
		test.InitNet() ;
		
		for(int m = 0 ; m < 10000 ; m++){
			System.out.println("Episode: " + m) ;
			float CurrMap[][] = test.GenerateMap() ;
			test.FrameBuffer = CurrMap ;
			int t = 0;
			float tReward = 0 ;
			//test.PrintBoard(CurrMap);
			for(int i = 0 ; i < 2*test.size ; i ++){
				//test.PrintBoard(CurrMap);
				int a = test.RLNet.GetAction(test.FlattenInput(t) , test.GetActionMask(CurrMap)) ;
				float NextMap[][] = test.DoMove(CurrMap, a) ;
				float r = test.CalcReward(CurrMap, NextMap) ;
				tReward += r;
				test.AddToBuffer(NextMap);
				t++;
				if(r == test.size*test.size + 1){
					test.RLNet.ObserveReward(r, null , test.GetActionMask(NextMap));
					break;
				}
				test.RLNet.ObserveReward(r, test.FlattenInput(t), test.GetActionMask(NextMap));
				CurrMap = NextMap;
				
			}
			//System.out.println("Net Score: " + (i));
		}
		
		Scanner keyboard = new Scanner(System.in);
				
		for(int m = 0 ; m < 100 ; m++){
			test.RLNet.SetEpsilon(0);
			float CurrMap[][] = test.GenerateMap();
			test.FrameBuffer = CurrMap;
			int t = 0;
			float tReward = 0;
			while(true){
				test.PrintBoard(CurrMap);
				keyboard.nextLine();
				int a = test.RLNet.GetAction(test.FlattenInput(t) , test.GetActionMask(CurrMap)) ;
				float NextMap[][] = test.DoMove(CurrMap, a) ;
				float r = test.CalcReward(CurrMap, NextMap) ;
				tReward += r;
				test.AddToBuffer(NextMap);
				t++;
				test.RLNet.ObserveReward(r, test.FlattenInput(t) , test.GetActionMask(NextMap));
				if(r == test.size*test.size + 1)
					break;
				CurrMap = NextMap;
			}
			System.out.println("Net Score: " + (tReward));
		}
		
		
		
	}

}
