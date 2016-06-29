package com.dap.dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Replay {

	INDArray Input;
	int Action; 
	float Reward;
	INDArray NextInput;
	int NextActionMask[] ;
	
	Replay(INDArray input , int action , float reward , INDArray nextInput , int nextActionMask[]){
		Input = input;
		Action = action;
		Reward = reward;
		NextInput = nextInput;
		NextActionMask = nextActionMask ;
	}
	
}
