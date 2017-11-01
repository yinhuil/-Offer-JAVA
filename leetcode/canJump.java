package leetcodeForJianzhi;
/*题目描述

Given an array of non-negative integers, you are initially positioned at the
first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.
For example:
A =[2,3,1,1,4], returntrue.
A =[3,2,1,0,4], returnfalse.*/



/*本题采用动态规划的方法，首先定一个跟A长度一样的boolean类型数组dp。
方法思想： 首先判断当前能否被前面跳到这个位置，如果可以的话，那就开始从当前位置的跳跃数往后进行遍历，
说明后面的这些步，都可以跳过去。 如果不可以的话，就break，说明后面更跳跃不过去了*/
public class canJump {
	public static boolean canJump(int[] A) {
		boolean[] dp=new boolean[A.length];
		dp[0]=true;
		for(int i=0;i<A.length-1;i++){
			if(dp[i]){
				for(int j=1;j<=A[i]&&(i+j)<A.length;j++){
					dp[i+j]=true;
				}
			}else{
				break;
			}
		}
		return dp[A.length-1];
	}
	public static void main(String[] args) {
		int[] a={3,2,1,0,4};
		System.out.println(canJump(a));
	}
}
