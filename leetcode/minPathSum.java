package leetcodeForJianzhi;
/*题目描述

Given a m x n grid filled with non-negative numbers, find a path from 
top left to bottom right which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.*/

//解析； 首先将第一行和第一列赋值，然后向下去最小
public class minPathSum {
	public int minPathSum(int[][] grid) {
		int[][] dp=new int[grid.length][grid[0].length];
		dp[0][0]=grid[0][0];
		for(int i=1;i<grid.length;i++){
			dp[i][0]=dp[i-1][0]+grid[i][0];
		}
		for(int j=1;j<grid[0].length;j++){
			dp[0][j]=dp[0][j-1]+grid[0][j];
		}
		for(int i=1;i<grid.length;i++){
			for(int j=1;j<grid[0].length;j++){
				dp[i][j]=Math.min(dp[i-1][j], dp[i][j-1])+grid[i][j];
			}
		}
		return dp[dp.length-1][dp[0].length-1];
	}
}
