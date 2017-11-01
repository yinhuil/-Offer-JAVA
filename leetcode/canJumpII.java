package leetcodeForJianzhi;
/*题目描述

Given an array of non-negative integers, you are initially positioned at the
first index of the array.Each element in the array represents your maximum 
jump length at that position.Your goal is to reach the last index in the minimum number 
of jumps.For example:Given array 
A =[2,3,1,1,4] The minimum number of jumps to reach the last index is2. 
(Jump1step from index 0 to 1, then3steps to the last index.)*/



/*解题思路：
 * 本题采用动态规划的思想解题思想：首先从第二个点出发，判断前面是否可以到达这个点，判断的依据是从前一个点start开始递减，发现A[start]是否>=(两个位置的差)，
如果发现则找最小值，反之，则返回-1，说明这个数组就不能到达最后。（最根本的思想，就是先考虑小数组再考虑大数组）*/
public class canJumpII {
	public static int jumpII(int[] A) {
		int[] dp=new int[A.length];
		for(int i=1;i<A.length;i++){
			int start=i-1;
			dp[i]=Integer.MAX_VALUE;
			for(;start>=0;start--){
				if(A[start]>=(i-start)){
					dp[i]=Math.min(dp[i], dp[start]+1);
				}
			}
			if(dp[i]==Integer.MAX_VALUE){
				return -1;
			}
		}
		return dp[A.length-1];
	}
	public static void main(String[] args) {
		int[] in={2,3,1,1,4};
		System.out.println(jumpII(in));
	}
}
