package leetcodeForJianzhi;

import java.util.Arrays;

/*题目描述

Given an array S of n integers, find three integers in S such that the sum is 
closest to a given number, target. Return the sum of the three integers. You may 
assume that each input would have exactly one solution.
    For example, given array S = {-1 2 1 -4}, and target = 1.

    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).*/
public class threeSumClosest {
//	用一个数表示第一个，然后后面两个表示后面的相加
	public static int threeSumClosest(int[] num, int target) {
		if(num.length<3) return -1;
		Arrays.sort(num);
		int result=num[0]+num[1]+num[2];
		for(int i=0;i<num.length;i++){
			int start=i+1,end=num.length-1;
			while(start<end){
				int temp=num[i]+num[start]+num[end];
				if(Math.abs(temp-target)<Math.abs(result-target)){
					result=temp;
				}
				if(temp>target) end--;
				else start++;
			}
			
		}
		return result;
	}
	public static void main(String[] args) {
		int[] in={1,1,-1,-1,3};
		System.out.println(threeSumClosest(in, -1));
		
	}
}
