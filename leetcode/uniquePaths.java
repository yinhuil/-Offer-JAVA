package leetcodeForJianzhi;

public class uniquePaths {
	
/*	A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
	The robot can only move either down or right at any point in time. The robot is trying to reach 
	the bottom-right corner of the grid (marked 'Finish' in the diagram below).
	How many possible unique paths are there?*/
	
//题解，本题强调的是，只能向下和向右来走，第一步是来把第一行和第一列定义为1，然后向下开始遍历，因为每一个点的路劲只能从上面来或者从左边来
//	所以就是左边和上面的和
	public int uniquePaths(int m, int n) {
		int[][] dp=new int[m][n];
		for(int i=0;i<m;i++){
			dp[i][0]=1;
		}
		for(int j=0;j<n;j++){
			dp[0][j]=1;
		}
		for(int i=1;i<m;i++){
			for(int j=1;j<n;j++){
				dp[i][j]=dp[i-1][j]+dp[i][j-1];
			}
		}
		return dp[m-1][n-1];
	}
}
