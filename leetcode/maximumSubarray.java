package leetcodeForJianzhi;
/*题目描述

Find the contiguous subarray within an array (containing at least one number) 
which has the largest sum.
For example, given the array[−2,1,−3,4,−1,2,1,−5,4],
the contiguous subarray[4,−1,2,1]has the largest sum =6.
click to show more practice.
More practice:
If you have figured out the O(n) solution, try coding another 
solution using the divide and conquer approach, which is more subtle.*/
public class maximumSubarray {
	public static int maxSubArray(int[] A) {
		int maxSum=0;
		int current=A[0];
		for(int i=0;i<A.length;i++){
			maxSum+=A[i];
			if(current<maxSum) current=maxSum;
			if(maxSum<0) maxSum=0;
		}
		return current;
		
	}
	public static void main(String[] args) {
		int[] in={-2,2,-1,4};
		System.out.println(maxSubArray(in));
	}
}
