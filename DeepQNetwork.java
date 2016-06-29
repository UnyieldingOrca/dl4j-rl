package com.dap.dl4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DeepQNetwork {

	int ReplayMemoryCapacity;
	List<Replay> ReplayMemory;
	double Epsilon;
	float Discount;
	MultiLayerNetwork DeepQ;
	MultiLayerNetwork TargetDeepQ;
	int BatchSize;
	int UpdateFreq;
	int UpdateCounter;
	int ReplayStartSize;
	Random r;
	
	int InputLength;
	int NumActions;
	
	INDArray LastInput;
	int LastAction;
	
	DeepQNetwork(MultiLayerConfiguration conf , int replayMemoryCapacity , float discount , 
			double epsilon , int batchSize , int updateFreq , int replayStartSize , int inputLength , int numActions){
		DeepQ = new MultiLayerNetwork(conf);
		DeepQ.init();
		TargetDeepQ = new MultiLayerNetwork(conf);
		TargetDeepQ.init();
		TargetDeepQ.setParams(DeepQ.params());
		ReplayMemoryCapacity = replayMemoryCapacity;
		Epsilon = epsilon;
		Discount = discount;
		r = new Random();
		BatchSize = batchSize;
		UpdateFreq = updateFreq;
		UpdateCounter = 0;
		ReplayMemory = new ArrayList<Replay>();
		ReplayStartSize = replayStartSize;
		InputLength = inputLength;
		NumActions = numActions;
	}
	
	void SetEpsilon(double e){
		Epsilon = e;
	}
	
	void AddReplay(float reward , INDArray NextInput , int NextActionMask[]){
		if( ReplayMemory.size() >= ReplayMemoryCapacity )
			ReplayMemory.remove( r.nextInt(ReplayMemory.size()) );
		ReplayMemory.add(new Replay(LastInput , LastAction , reward , NextInput , NextActionMask));
	}
	
	Replay[] GetMiniBatch(int BatchSize){
		int size = ReplayMemory.size() < BatchSize ? ReplayMemory.size() : BatchSize ;
		Replay[] retVal = new Replay[size];
		
		for(int i = 0 ; i < size ; i++){
			retVal[i] = ReplayMemory.get(r.nextInt(ReplayMemory.size()));
		}
		return retVal;
		
	}
	
	float FindMax(INDArray NetOutputs , int ActionMask[]){
		int i = 0;
		while(ActionMask[i] == 0) i++;
		
		float maxVal = NetOutputs.getFloat(i);
		for(; i < NetOutputs.size(1) ; i++){
			if(NetOutputs.getFloat(i) > maxVal && ActionMask[i] == 1){
				maxVal = NetOutputs.getFloat(i);
			}
		}
		return maxVal;
	}
	
	int FindActionMax(INDArray NetOutputs , int ActionMask[]){
		int i = 0;
		while(ActionMask[i] == 0) i++;
		
		float maxVal = NetOutputs.getFloat(i);
		int maxValI = i;
		for(; i < NetOutputs.size(1) ; i++){
			if(NetOutputs.getFloat(i) > maxVal && ActionMask[i] == 1){
				maxVal = NetOutputs.getFloat(i);
				maxValI = i;
			}
		}
		return maxValI;
	}
	
	
	int GetAction(INDArray Inputs , int ActionMask[]){
		LastInput = Inputs;
		INDArray outputs = DeepQ.output(Inputs);
		System.out.print(outputs + " ");
		if(Epsilon > r.nextDouble()) {
			 LastAction = r.nextInt(outputs.size(1));
			 while(ActionMask[LastAction] == 0)
				 LastAction = r.nextInt(outputs.size(1));
			 System.out.println(LastAction);
			 return LastAction;
		}
		
		LastAction = FindActionMax(outputs , ActionMask);
		System.out.println(LastAction);
		return LastAction;
	}
	
	void ObserveReward(float Reward , INDArray NextInputs , int NextActionMask[]){
		AddReplay(Reward , NextInputs , NextActionMask);
		if(ReplayStartSize <  ReplayMemory.size())
			TrainNetwork(BatchSize);
		UpdateCounter++;
		if(UpdateCounter == UpdateFreq){
			UpdateCounter = 0;
			System.out.println("Reconciling Networks");
			ReconcileNetworks();
		}
	}
	
	INDArray CombineInputs(Replay replays[]){
		INDArray retVal = Nd4j.create(replays.length , InputLength);
		for(int i = 0; i < replays.length ; i++){
			retVal.putRow(i, replays[i].Input);
		}
		return retVal;
	}
	
	INDArray CombineNextInputs(Replay replays[]){
		INDArray retVal = Nd4j.create(replays.length , InputLength);
		for(int i = 0; i < replays.length ; i++){
			if(replays[i].NextInput != null)
				retVal.putRow(i, replays[i].NextInput);
		}
		return retVal;
	}
	void TrainNetwork(int BatchSize){
		Replay replays[] = GetMiniBatch(BatchSize);
		INDArray CurrInputs = CombineInputs(replays);
		INDArray TargetInputs = CombineNextInputs(replays);

		float TotalError = 0;
		
		INDArray CurrOutputs = DeepQ.output(CurrInputs);
		INDArray TargetOutputs = TargetDeepQ.output(TargetInputs);
		float y[] = new float[replays.length];
		for(int i = 0 ; i < y.length ; i++){
			int ind[] = { i , replays[i].Action };
			float FutureReward = 0 ;
			if(replays[i].NextInput != null)
				FutureReward = FindMax(TargetOutputs.getRow(i) , replays[i].NextActionMask);
			float TargetReward = replays[i].Reward + Discount * FutureReward ;
			TotalError += (TargetReward - CurrOutputs.getFloat(ind)) * (TargetReward - CurrOutputs.getFloat(ind));
			CurrOutputs.putScalar(ind , TargetReward ) ;
		}
		//System.out.println("Avgerage Error: " + (TotalError / y.length) );
		
		DeepQ.fit(CurrInputs, CurrOutputs);
	}
	
	void ReconcileNetworks(){
		TargetDeepQ.setParams(DeepQ.params());
	}
	
	public boolean SaveNetwork(String ParamFileName , String JSONFileName){
	    //Write the network parameters:
	    try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(ParamFileName)))){
	        Nd4j.write(DeepQ.params(),dos);
	    } catch (IOException e) {
	    	System.out.println("Failed to write params");
			return false;
		}
	    
	    //Write the network configuration:
	    try {
			FileUtils.write(new File(JSONFileName), DeepQ.getLayerWiseConfigurations().toJson());
		} catch (IOException e) {
			System.out.println("Failed to write json");
			return false;
		}
	    return true;
	}
	
	public boolean LoadNetwork(String ParamFileName , String JSONFileName){
		//Load network configuration from disk:
	    MultiLayerConfiguration confFromJson;
		try {
			confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(JSONFileName)));
		} catch (IOException e1) {
			System.out.println("Failed to load json");
			return false;
		}

	    //Load parameters from disk:
	    INDArray newParams;
	    try(DataInputStream dis = new DataInputStream(new FileInputStream(ParamFileName))){
	        newParams = Nd4j.read(dis);
	    } catch (FileNotFoundException e) {
	    	System.out.println("Failed to load parems");
			return false;
		} catch (IOException e) {
	    	System.out.println("Failed to load parems");
			return false;
		}
	    //Create a MultiLayerNetwork from the saved configuration and parameters 
	    DeepQ = new MultiLayerNetwork(confFromJson); 
	    DeepQ.init(); 
	    DeepQ.setParameters(newParams); 
	    ReconcileNetworks();
	    return true;
	    
	}
	
	
}
