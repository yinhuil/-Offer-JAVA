package leetcodeForJianzhi;
/*题目描述

There are N gas stations along a circular route, where the amount of gas at
station i isgas[i].You have a car with an unlimited gas tank and it costscost[i]of 
gas to travel from station i to its next station (i+1). You begin the journey with 
an empty tank at one of the gas stations.Return the starting gas station's index 
if you can travel around the circuit once, otherwise return -1.
Note: 
The solution is guaranteed to be unique.*/

//理解：这道题很棒：   题意是选择从哪个加油站开始算起，最后可以走完这个循环。
//    首先选择从最后一个开始，如果此时sum>=0，则可以继续从end=0开始算起，
// 	  如果此时的sum<0，相当于第一个位置不行，应该从倒数第二个开始。。。。。 
//   最后判断sum的值，如果>=0，出发位置就是start，反之就是不行，即-1
public class GasStation {
	public int canCompleteCircuit(int[] gas, int[] cost) {
		int start=gas.length-1;
		int end=0;
		int sum=0;
		sum+=gas[start]-cost[start];
		while(start>end){
			if(sum>0){
				sum+=gas[end]-cost[end];
				end++;
			}else{
				start--;
				sum+=gas[start]-cost[start];
			}
		}
		return sum>=0?start:-1;
	}
}
